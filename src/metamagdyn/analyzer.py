# -*- coding: utf-8 -*-

"""
MetaMagDyn

Author: Diego González Chávez
email : diegogch@cbpf.br / diego.gonzalez.chavez@gmail.com
"""

import os
import numpy
import functools
import time
import textwrap
import scipy.interpolate
import matplotlib.pyplot as plt

from . import runner
from . import OVF_V2_0


def area_MxB(ms, bs, closed=False):
    # http://en.wikipedia.org/wiki/Polygon#Area_and_centroid
    area = 0
    for i in range(len(ms)-1):
        area += ms[i]*bs[i+1] - ms[i+1]*bs[i]
    if not(closed):
        area += ms[-1]*bs[0] - ms[0]*bs[-1]
    area = 0.5 * numpy.abs(area)
    return numpy.atleast_1d(area).mean()


def lockin(signal, ts, f, refPhase,
           ref_path='i...,i...->i...',
           signal_path='i...,i...->...'):
    '''
    Calculates the amplitude and phase of a signal
    '''
    angle = 2*numpy.pi*numpy.einsum(ref_path, ts, numpy.atleast_1d(f))
    sR, cR = numpy.sin(angle), numpy.cos(angle)

    data_aS = 2*numpy.einsum(signal_path, signal, sR)/sR.shape[0]
    data_aC = 2*numpy.einsum(signal_path, signal, cR)/cR.shape[0]
    amp = (data_aS**2 + data_aC**2)**0.5
    phase = numpy.arctan2(data_aC, data_aS) - refPhase
    return amp, phase


class SimAnalyzer:

    def __init__(self, name, inMem=False, gpu=0):
        '''
        SimAnalyzer object

        Params:
            name : Simulation directory
            inMem : True : data is loaded fron Sim_Tables file
                    False : already simulated data is loaded in memory
        '''
        self.name = name
        self.runner = runner.SimRunner(self.name, default_gpu=gpu)

        self._SimTables = numpy.load(self.runner._Sim_Tables_path,
                                     allow_pickle=True,
                                     mmap_mode='r')

        self.inMem = inMem
        if self.inMem:
            self._SimTables = self._SimTables.copy()

        ArrangedSimTable = numpy.einsum('hftd->dhft', self._SimTables)

        self._tOsc = ArrangedSimTable[0][:, :, 1:]
        self._mOsc = ArrangedSimTable[1:4][:, :, :, 1:]
        self._bOsc = ArrangedSimTable[4:7][:, :, :, 1:]

        eqTable = numpy.atleast_2d(numpy.loadtxt(
            f'{self.runner._Eq_Data_path}/table.txt'))
        eqTable = numpy.einsum('ij->ji', eqTable)

        self._mStatic = eqTable[1:4]

        if self.runner._ExtraTableAdds > 0:
            self.Extra_Tables = numpy.load(self.runner._Extra_Tables_path,
                                           allow_pickle=True,
                                           mmap_mode='r+')
            self.ExtraData = numpy.einsum('hftd->dhft',
                                          self.Extra_Tables)[:, :, :, 1:]

    def __repr__(self):
        return f'''{self.name} SimAnalyzer'''

    def getFieldAnalyzer(self, h):
        return Static_Data(self, h)

    def fi(self, f):
        '''
        Returns fs index correspondig to frequency f
            f = self.fs[index]
        '''
        return numpy.argmin(numpy.abs(self.fs - f))

    def hi(self, h):
        '''
        Returns hs index correspondig to DC field h
            h = self.hs[index]
        '''
        return numpy.argmin(numpy.abs(self.hs - h))

    @property
    def hs(self):
        '''
        List of simulated DC fields
        '''
        return self.runner._hs

    @property
    def fs(self):
        '''
        List of simulated frequencies
        '''
        return self.runner._fs

    @property
    def doneMask(self):
        '''
        Boolean array of allready simulated {hi, fi} points
        .doneMask [hi, fi]
        '''
        return self._SimTables[:, :, 0, 0] > 0

    @property
    def DC_dir(self):
        '''
        DC field direction vector
        '''
        return self.runner._DC_dir

    @property
    def RF_dir(self):
        '''
        RF field direction vector
        '''
        return self.runner._RF_dir

    @property
    def tOsc(self):
        '''
        Array of simulation times (between one oscilition period)
        for all fields and frequecies
            .tOsc [hi, fi, ti]
        '''
        return self._tOsc

    @property
    def mOsc(self):
        '''
        Array of spatial average magnetization direction
        for all fields and frequecies and simulation times
            .mOsc [xyz, hi, fi, ti]
        '''
        return self._mOsc

    @property
    def bOsc(self):
        '''
        Array of spatial average field DC + RF (in Tesla)
        for all fields and frequecies and simulation times
            .bOsc [xyz, hi, fi, ti]
        '''
        return self._bOsc

    @property
    def mStatic(self):
        '''
        Vector list of magnetization spatial average for all DC fields
            .mStatic [xyz, hi]
        '''
        return self._mStatic

    @functools.cached_property
    def absorbed_power(self):
        '''
        Array of Microwave Absorbed Power (interpolated)
        for all fields and frequecies
            .absorbed_power [hi, fi]
        '''
        try:
            nh, nf = self.raw_absorbed_power.shape
            X, Y = numpy.meshgrid(range(nf), range(nh))
            pts = numpy.array([X[self.doneMask], Y[self.doneMask]]).T
            values = self.raw_absorbed_power[self.doneMask]
            AbsPow = scipy.interpolate.griddata(pts, values,
                                                (X, Y),
                                                method='linear')
            AbsPow[numpy.isnan(AbsPow)] = 0
            return AbsPow
        except:
            return numpy.zeros_like(self.raw_absorbed_power)

    @functools.cached_property
    def raw_absorbed_power(self):
        '''
        Array of Microwave Absorbed Power (not interpolated)
        for all fields and frequecies
            .raw_absorbed_power [hi, fi]
        '''

        m_proy = numpy.einsum('ijkl,i->jkl', self._mOsc, self.RF_dir)
        h_proy = numpy.einsum('ijkl,i->jkl', self._bOsc, self.RF_dir)
        nh, nf, nc = self._tOsc.shape
        AbsPow = numpy.zeros((nh, nf))
        # TODO Parallelize this
        for hi in range(nh):
            for fi in range(nf):
                AbsPow[hi, fi] = area_MxB(m_proy[hi, fi],
                                          h_proy[hi, fi]) * self.fs[fi]
        AbsPow[~self.doneMask] = numpy.nan
        return AbsPow


class Static_Data:
    '''Static (DC-Field) Data Holder for SimAnalyzer'''

    def __init__(self, SimAnalyzer_parent, h):
        '''
        Static (DC-Field) Data Holder init for SimAnalyzer

        Params:
            SimAnalyzer_parent : SimAnalyzer object
            h : Simulation field
        '''
        self._SymAnalyzer = SimAnalyzer_parent
        self._LoadField(h)

    def __repr__(self):
        return textwrap.dedent(f'''\
        Static (DC) Simulation data holder for {self._SymAnalyzer.name:s} SimAnalyzer
         Field = {self.h:0.4f}
          hi = {self._hi}\
        ''')

    def getFrequencyAnalyzer(self, f):
        return Oscillation_Data(self, f)

    @property
    def h(self):
        '''
        DC Field of this object
        '''
        return self._SymAnalyzer.hs[self._hi]

    @property
    def vector_field(self):
        '''
        Static Magnetization Vector Field (OMMF_Vector_Field class)
            It correspond to DC field 'H'
            used in .LoadField(H)

        OMMF_Vector_Field class members:
            t : simulation time
            xnodes : number of x nodes
            ynodes : number of y nodes
            znodes : number of z nodes
            dx : distance between x nodes
            dy : distance between x nodes
            dz : distance between x nodes
            data : Array of vector field data [xi, yi, zi, xyz]
            vx : Array of x component = data[:,:,:,0]
            vy : Array of y component = data[:,:,:,1]
            vz : Array of z component = data[:,:,:,2]
        '''
        return self._vf

    @property
    def nan_mask(self):
        '''
        XY Mask were Vector Fields data is not valid
            .nan_mask  [xi, yi]
        '''
        return self._nan_mask

    @property
    def u_r(self):
        '''
        Direction of static (DC) magnetization (unitary vectors)
        for each grid node
            .u_r [xi, yi, zi, xyz]
        '''
        return self._u_r

    @property
    def u_phi(self):
        '''
        In plane vectors perpendicular to the static (DC) magnetization
        (unitary vectors) for each grid node
            .u_phi [xi, yi, zi, xyz]
        '''
        return self._u_phi

    @property
    def u_theta(self):
        '''
        Out plane vectors perpendicular to the static (DC) magnetization
        (unitary vectors) for each grid node
            .u_theta [xi, yi, zi, xyz]
        '''
        return self._u_theta

    @property
    def m_amp(self):
        '''
        Spatial mean magnetization oscillation amplitude vector
        for all frequencies
            .m_amp [xyz, fi]
        '''
        return self._m_amp

    @property
    def m_phase(self):
        '''
        Phases (in rads) of the spatial mean magnetization oscillation,
        with respect to the RF field, for all frequencies
            .m_phase [xyz, fi]
        '''
        return self._m_phase

    @property
    def m_amp2f(self):
        '''
        Spatial mean magnetization oscillation second harmonic
        amplitude vector for all frequencies
            .m_amp2f [xyz, fi]
        '''
        return self._m_amp2f

    def _LoadField(self, h):
        '''
        ._LoadField(h)

        Loads the static vector field corresponding to DC field h
        self.vf_Eq  -> OMMF_Vector_Field class

        and calculates:
        The not valid data mask (m = 0):
        self.nan_mask  [xi, yi]

        The unitary vectors
        self.u_r       [xi, yi, zi, xyz] (direction of magnetization)
        self.u_phi     [xi, yi, zi, xyz] (in plane and perpendicular to u_r)
        self.u_theta   [xi, yi, zi, xyz] (out of plane and perpendicular to u_r)

        The phase and amplitude of the spatial mean magnetization
        (for all the frequencies)
        self.m_amp      [xyz, fi]
        self.m_phase    [xyz, fi]

        self.m_amp2f    [xyz, fi] (amplitude in the second harmonic)
        '''
        hi = self._SymAnalyzer.hi(h)
        self._hi = hi

        my_runner = self._SymAnalyzer.runner
        eqOVF_Path = os.path.join(my_runner._Eq_Data_path, f'Eq_hi{hi:d}.ovf')
        self._vf = OVF_V2_0.load_OOMMF_VectorField(eqOVF_Path)
        self._nan_mask = (numpy.abs(self._vf.data).sum(axis=(2, 3)) == 0).T

        self._u_r = self._vf.data.copy()
        self._u_phi = numpy.cross(numpy.array(
            [0, 0, 1])[None, None, None, :], self._u_r, axis=-1)
        self._u_theta = numpy.cross(self._u_r, self._u_phi, axis=-1)

        norm_u_phi = numpy.linalg.norm(self._u_phi, axis=-1)[:, :, :, None]
        norm_u_phi[norm_u_phi == 0] = -1
        norm_u_theta = numpy.linalg.norm(self._u_theta, axis=-1)[:, :, :, None]
        norm_u_theta[norm_u_theta == 0] = -1

        self._u_phi = self._u_phi/norm_u_phi
        self._u_theta = self._u_theta/norm_u_theta
        self._u_phi[self._nan_mask.T] = 0
        self._u_theta[self._nan_mask.T] = 0

        # We calculate amplitude and phase of <m> and b
        ts = self._SymAnalyzer.tOsc[hi, :, :]
        fs = self._SymAnalyzer.fs

        hrf_abs = numpy.einsum('dft,d->tf',
                               self._SymAnalyzer.bOsc[:, hi, :, :],
                               self._SymAnalyzer.RF_dir)

        rf_amp, rf_phase = lockin(hrf_abs, ts, fs, 0, ref_path='ft,f->tf')

        # Phase of the recorded RF field for al frequencies [fi]
        self._refPhase = rf_phase

        m = self._SymAnalyzer.mOsc[:, hi, :, :]

        self._m_amp, self._m_phase = lockin(m, ts, fs, self._refPhase,
                                            ref_path='ft,f->tf',
                                            signal_path='dft,tf->df')

        # 2f components
        self._m_amp2f, _phase = lockin(m, ts, 2*fs, self._refPhase,
                                       ref_path='ft,f->tf',
                                       signal_path='dft,tf->df')

    def GetStaticField(self, field_name):
        my_runner = self._SymAnalyzer.runner
        my_runner.Run_StaticField(hi=self._hi,
                                  field_name=field_name,
                                  file_name='StaticField.mx3')
        SF_Path = os.path.join(my_runner._path,
                               'StaticField.out/StaticField.ovf')
        return OVF_V2_0.load_OOMMF_VectorField(SF_Path).data


class Oscillation_Data:
    '''Oscillation (RF) Data Holder for SimAnalyzer'''

    def __init__(self, StaticData_parent, f):
        '''
        Oscillation (RF) Data Holder init for SimAnalyzer

        Params:
            parent : FieldAnalyzer object
            f : simulation frequency
        '''
        self._SymAnalyzer = StaticData_parent._SymAnalyzer
        self._StaticData = StaticData_parent
        self._LoadFrequency(f)

    def __repr__(self):
        return textwrap.dedent(f'''\
        Oscillation (RF) data holder for {self._SymAnalyzer.name} SimAnalyzer
         Field = {self._StaticData.h:0.4f}
          hi = {self._StaticData._hi}
         Frequency = {self.f/1E9:0.4f} GHz
          fi ={self._fi}\
        ''')

    @property
    def f(self):
        '''
        Frquency of this object
        '''
        return self._SymAnalyzer.fs[self._fi]

    @property
    def vector_fields(self):
        '''
        List of Magnetization Vector Field snapshots
        along one oscilation period
        (List OMMF_Vector_Field classes)
            It correspond to frequencie 'f'
            used in .LoadFrequency(f)

        OMMF_Vector_Field class members:
            t : simulation time
            xnodes : number of x nodes
            ynodes : number of y nodes
            znodes : number of z nodes
            dx : distance between x nodes
            dy : distance between x nodes
            dz : distance between x nodes
            data : Array of vector field data [xi, yi, zi, xyz]
            vx : Array of x component = data[:,:,:,0]
            vy : Array of y component = data[:,:,:,1]
            vz : Array of z component = data[:,:,:,2]
        '''
        return self._vfs

    @property
    def raw_data(self):
        '''
        Magnetization direction vectors for
        all times along one oscillation period,
        and for all grid nodes
            .raw_data [ti, xi, yi, zi, xyz]
        '''
        return self._raw_data

    @property
    def m_amp(self):
        '''
        Oscillation amplitude of the magnetization vector,
        for each grid node
            .m_amp [xi, yi, zi, xyz]
        '''
        return self._m_amp

    @property
    def m_phase(self):
        '''
        Phases of each component of the oscillating
        magnetization vector, for each grid node
            .m_phase [xi, yi, zi, xyz]
        '''
        return self._m_phase

    @property
    def phi_amp(self):
        '''
        Oscillation amplitude of the in-plane component of the
        magnetization vector, for each grid node
            .phi_amp [xi, yi, zi]
        '''
        return self._phi_amp

    @property
    def phi_phase(self):
        '''
        Phase of the in-plane component of the oscillating
        magnetization vector, for each grid node
            .phi_phase [xi, yi, zi]
        '''
        return self._phi_phase

    @property
    def theta_amp(self):
        '''
        Oscillation amplitude of the out-of-plane component of the
        magnetization vector, for each grid node
            .theta_amp [xi, yi, zi]
        '''
        return self._theta_amp

    @property
    def theta_phase(self):
        '''
        Phase of the out-of-plane component of the oscillating
        magnetization vector, for each grid node
            .theta_phase [xi, yi, zi]
        '''
        return self._theta_phase

    @property
    def phi_amp_2f(self):
        '''
        Oscillation amplitude of the second harmonic of the in-plane
        component of the magnetization vector, for each grid node
            .phi_amp_2f [xi, yi, zi]
        '''
        return self._phi_amp_2f

    @property
    def phi_phase_2f(self):
        '''
        Second harmonic phase of the in-plane component of the
        oscillating  magnetization vector, for each grid node
            .phi_phase_2f [xi, yi, zi]
        '''
        return self._phi_phase_2f

    @property
    def theta_amp_2f(self):
        '''
        Oscillation amplitude of the second harmonic of the out-of-plane
        component of the magnetization vector, for each grid node
            .theta_amp_2f [xi, yi, zi]
        '''
        return self._theta_amp_2f

    @property
    def theta_phase_2f(self):
        '''
        Second harmonic phase of the "out-of-plane" component of the
        oscillating  magnetization vector, for each grid node
            .theta_phase_2f [xi, yi, zi]
        '''
        return self._phi_phase_2f

    def _LoadFrequency(self, f):
        '''
        Load the oscillation vector field corresponding to frequency f

        self.vector_fields  [ti] -> list of  OMMF_Vector_Field class
        self.raw_data       [ti, xi, yi, zi, xyz]  -> magnetization direction

        and calculates:
            .m_amp    [xi, yi, zi, xyz]
            .m_phase  [xi, yi, zi, xyz]

            .phi_amp   [xi, yi, zi]
            .phi_phase [xi, yi, zi]

            .theta_amp   [xi, yi, zi]
            .theta_phase [xi, yi, zi]

            .phi_amp_2f   [xi, yi, zi]
            .phi_phase_2f [xi, yi, zi]

            .theta_amp_2f   [xi, yi, zi]
            .theta_phase_2f [xi, yi, zi]
        '''

        my_runner = self._SymAnalyzer.runner
        SymAnalyzer = self._SymAnalyzer
        StaticData = self._StaticData

        hi = StaticData._hi
        fi = SymAnalyzer.fi(f)

        my_runner.Run_Modes(hi, fi)
        vf_Oscs = []
        oscFiles_Path = os.path.join(my_runner._path, 'Mode.out/')
        for ci in range(SymAnalyzer.tOsc.shape[2]):
            vf_Osc = OVF_V2_0.load_OOMMF_VectorField(
                f'{oscFiles_Path}Osc_ci{ci}.ovf')
            vf_Oscs.append(vf_Osc)
        self._vfs = vf_Oscs
        data = numpy.array([vf.data for vf in self._vfs])
        # TODO ???
        self._raw_data = data.copy()
        data -= data.mean(axis=0)[None, :]

        self._fi = fi
        self._hi = hi

        # Calculate amplitude and phase of the xyz magnetization vectors
        ts = SymAnalyzer.tOsc[hi, fi, :]

        # drop ts > ts[0] + T
        ts_ok = (ts < ts[0] + 1/self.f)
        ts = ts[ts_ok]
        data = data[ts_ok, :, :, :, :]

        refPhase = StaticData._refPhase[fi]
        # Phase of the recorded RF field for this frequency
        self._refPhaseOsc = refPhase

        self._m_amp, self._m_phase = lockin(data, ts, self.f, refPhase)
        # shape = [xi, yi, zi, xyz]

        # Projections
        data_u_phi = numpy.einsum('txyzi,xyzi->txyz',
                                  data, StaticData.u_phi)
        data_u_theta = numpy.einsum('txyzi,xyzi->txyz',
                                    data, StaticData.u_theta)

        self._phi_amp, self._phi_phase = lockin(
            data_u_phi, ts, self.f, refPhase)
        # shape [xi, yi, zi]

        self._theta_amp, self._theta_phase = lockin(
            data_u_theta, ts, self.f, refPhase)
        # shape [xi, yi, zi]

        # Calculations for 2f

        self._phi_amp_2f, self._phi_phase_2f = lockin(
            data_u_phi, ts, 2*self.f, refPhase)
        # shape [xi, yi, zi]

        self._theta_amp_2f, self._theta_phase_2f = lockin(
            data_u_theta, ts, 2*self.f, refPhase)
        # shape [xi, yi, zi]

        # nan = 0
        for amp_array in [self._m_amp,
                          self._phi_amp, self._theta_amp,
                          self._phi_amp_2f, self._theta_amp_2f]:
            amp_array[numpy.isnan(amp_array)] = 0
        # Acotamos la fase
        for phs_array in [self._m_phase,
                          self._phi_phase, self._theta_phase,
                          self._phi_phase_2f, self._theta_phase_2f]:
            phs_array[phs_array > numpy.pi] -= 2 * numpy.pi
            phs_array[phs_array < -numpy.pi] += 2 * numpy.pi

    def GetDynamicField(self, field_name):
        my_runner = self._SymAnalyzer.runner
        my_runner.Run_DynamicField(hi=self._hi,
                                   fi=self._fi,
                                   field_name=field_name,
                                   file_name='DynamicField.mx3')
        vf_Oscs = []
        DF_Path = os.path.join(my_runner._path, 'DynamicField.out/')
        for ci in range(self._SymAnalyzer.tOsc.shape[2]):
            vf_Osc = OVF_V2_0.load_OOMMF_VectorField(
                f'{DF_Path}DynField_ci{ci}.ovf')
            vf_Oscs.append(vf_Osc)
        return numpy.array([vf.data for vf in vf_Oscs])
