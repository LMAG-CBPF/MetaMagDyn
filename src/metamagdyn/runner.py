# -*- coding: utf-8 -*-

"""
MetaMagDyn

Author: Diego González Chávez
email : diegogch@cbpf.br / diego.gonzalez.chavez@gmail.com
"""

import os
import shutil
import subprocess
import threading
import time

import numpy
import pkg_resources
from . import threading_decorators as ThD


def join(*paths):
    return os.path.join(*paths).replace('\\', '/')


def _get_file_as_string(fileName):
    return pkg_resources.resource_string(__name__, fileName).decode('ascii')


_mx3_defs = _get_file_as_string('MumaxSrc/Defs_base.mx3')
_mx3_eq = _get_file_as_string('MumaxSrc/Eq_base.mx3')
_mx3_osc = _get_file_as_string('MumaxSrc/Osc_base.mx3')
_mx3_modes = _get_file_as_string('MumaxSrc/Modes_base.mx3')
_mx3_dynamic_field = _get_file_as_string('MumaxSrc/DynamicField_base.mx3')
_mx3_static_field = _get_file_as_string('MumaxSrc/StaticField_base.mx3')

mumaxBin = "/usr/bin/mumax3"


def run_mumax3(mx3_file, kernel_path=None, gpu=None):
    if gpu is None:
        gpu = 0
    if kernel_path is None:
        kernel_path = "/tmp"

    path = os.path.dirname(os.path.abspath(mx3_file))
    out_file = join(path, 'pythonMumax_stdout.txt')
    err_file = join(path, 'pythonMumax_stderr.txt')
    with open(out_file, 'w') as f_out, open(err_file, 'w') as f_err:
        subprocess.call([mumaxBin,
                         '-i=false',
                         f'-cache={kernel_path}',
                         f'-gpu={gpu}',
                         mx3_file],
                        stdout=f_out,
                        stderr=f_err
                        )


class SimRunner:
    def __init__(self,
                 name_or_mx3_file,
                 hs=None,
                 fs=None,
                 DC_dir=None,
                 RF_dir=None,
                 TableAdds=0,
                 kernel_path=None,
                 default_gpu=0):
        '''
        Creates a SimRunner object

        Parameters:
        ----------

        name_or_mx3_file : .mx3 file or directory name
            If .mx3 file is used a new SimRunner definition will be created
            else, the definition is loaded from the directory
        hs : List DC magnetic fields
        fs : List of frequencies
        DC_dir : DC field direction in [x,y,z] coordinates
        RF_dir : RF field direction in [x,y,z] coordinates
        TableAdds : Number of TableAdd() calls, besides TableAdd(B_ext),
                    in the .mx3 code
        kernel_path : Cache path (for kernel) passed to mumax3 
        '''

        self._running_threads = []
        self._Semaphore = threading.Semaphore()
        self._gpu = default_gpu
        self._stop = False

        self._hs = hs
        self._fs = fs
        self._DC_dir = DC_dir
        self._RF_dir = RF_dir

        self._ExtraTableAdds = TableAdds
        self._last_mode = None

        if os.path.isdir(name_or_mx3_file):
            self._load(name_or_mx3_file)
        else:
            # New file
            with open(name_or_mx3_file, 'r') as myfile:
                self._mx3_base = myfile.read()
            # Default mx3 files
            self._mx3_defs = _mx3_defs
            self._mx3_eq = _mx3_eq
            self._mx3_osc = _mx3_osc
            self._mx3_modes = _mx3_modes

        if kernel_path is None:
            self.kernel_path = "/tmp"
        else:
            if not os.path.isdir(kernel_path):
                os.makedirs(kernel_path)
            self.kernel_path = os.path.abspath(kernel_path)

    def save(self, sim_name, overwrite=False):
        '''
        Save the SimRunner definition into a directory

        Parameters:
        ----------

        sim_name : Simulation name (Directory name)
        overwrite : Whether to overwrite an existing simulation or not
        '''

        # If not all data is filled, then print missing data and return.
        if True in [x is None for x in self.Info.values()]:
            print('Not saved')
            print('Please fill all needed data')
            for k, v in self.Info.items():
                if v is None:
                    print(k, 'is None')
            return

        if not os.path.isdir(sim_name):
            os.makedirs(sim_name)
        elif overwrite:
            shutil.rmtree(sim_name)  # Delete the directory
            os.makedirs(sim_name)

        self._path = os.path.abspath(sim_name)

        self._OscData_path = join(self._path, 'OscData')
        if not os.path.isdir(self._OscData_path):
            os.makedirs(self._OscData_path)

        # Sim info
        Sim_info_file = join(self._path, 'Sim_info.npy')
        if os.path.exists(Sim_info_file):
            print('Sim_info already exists')
        else:
            numpy.save(Sim_info_file, self.Info)

        # Sim_Tables
        self._Sim_Tables_path = join(self._path, 'Sim_tables.npy')
        if os.path.exists(self._Sim_Tables_path):
            print('Sim_tables already exists')
        else:
            shape = (len(self._hs), len(self._fs), 21, 7)
            Sim_Tables = numpy.zeros(shape)
            numpy.save(self._Sim_Tables_path, Sim_Tables)

        # Extra_Tables colums
        if self._ExtraTableAdds > 0:
            self._Extra_Tables_path = join(self._path, 'Extra_tables.npy')
            if os.path.exists(self._Extra_Tables_path):
                print('Extra_tables already exists')
            else:
                shape = (len(self._hs), len(self._fs),
                         21, self._ExtraTableAdds)
                Extra_Tables = numpy.zeros(shape)
                numpy.save(self._Extra_Tables_path, Extra_Tables)

    def _load(self, sim_name):
        self._path = os.path.abspath(sim_name)
        _sim_info_file = join(self._path, 'Sim_info.npy')
        data = numpy.load(_sim_info_file, allow_pickle=True).item()
        self._mx3_base = data['mx3_base']
        self._mx3_defs = data['mx3_defs']
        self._mx3_eq = data['mx3_eq']
        self._mx3_osc = data['mx3_osc']
        self._mx3_modes = data['mx3_modes']
        self._hs = data['h']
        self._fs = data['f']
        self._DC_dir = data['DC_dir']
        self._RF_dir = data['RF_dir']
        self._OscData_path = join(self._path, 'OscData')
        self._Sim_Tables_path = join(self._path, 'Sim_tables.npy')
        self._Eq_Data_path = join(self._path, 'DC_sweep.out')
        self._Modes_path = join(self._path, 'Mode.out')

        if 'ExtraTableColumns' in data.keys():
            self._Extra_Tables_path = join(self._path, 'Extra_tables.npy')
            self._ExtraTableAdds = data['ExtraTableColumns']
        else:
            self._ExtraTableAdds = 0

    def _create_mx3_file(self, file_name, template, data):
        _mx3_file = join(self._path, file_name)
        with open(_mx3_file, 'w') as file:
            file.write(self._mx3_base)
            file.write(self._mx3_defs)
            if isinstance(data, dict):
                data = [data]
            for datum in data:
                file.write(template.format(**datum))
        return _mx3_file

    def Run_DC_Field(self, gpu=None):
        if gpu is None:
            gpu = self._gpu
        if self.status != 'Idle':
            return

        DC_data = [{'h': h,
                    'hi': hi,
                    'DC_dir': tuple(self._DC_dir)
                    } for hi, h in enumerate(self._hs)]
        _mx3_file = self._create_mx3_file(file_name='DC_sweep.mx3',
                                          template=self._mx3_eq,
                                          data=DC_data)
        DC_Runner = ThD.as_thread(run_mumax3)
        DC_Runner.info = 'DC_Runner'
        DC_Runner.mx3_file = 'DC_sweep.mx3'
        DC_Runner(_mx3_file, kernel_path=self.kernel_path, gpu=gpu)
        self._running_threads.append(DC_Runner)

    def Run_Osc(self, hi, fi, file_name='Osc.mx3', gpu=None):
        if gpu is None:
            gpu = self._gpu
        if 'DC_Runner' in self.status:
            return
        if file_name in [th.mx3_file for th in self._running_threads]:
            return
        if self._hi_fi_Done(hi, fi):
            return

        osc_data = {'DC_state': join(self._path, 'DC_sweep.out',
                                     f'Eq_hi{hi}.ovf'),
                    'h_dc': self._hs[hi],
                    'freq': self._fs[fi],
                    'DC_dir': tuple(self._DC_dir),
                    'RF_dir': tuple(self._RF_dir)
                    }
        _mx3_file = self._create_mx3_file(file_name=file_name,
                                          template=self._mx3_osc,
                                          data=osc_data)
        sim_dir = f'{os.path.splitext(_mx3_file)[0]}.out/'

        def _osc_runner():
            run_mumax3(_mx3_file, kernel_path=self.kernel_path, gpu=gpu)
            self._save_data_hi_fi(hi, fi, sim_dir)
        Osc_Runner = ThD.as_thread(_osc_runner)
        Osc_Runner.info = ('Osc', hi, fi)
        Osc_Runner.mx3_file = file_name
        Osc_Runner()
        self._running_threads.append(Osc_Runner)

    def Run_All_Osc(self, n_threads=2, indices='Auto', gpu=None):
        if gpu is None:
            gpu = self._gpu
        if 'DC_Runner' in self.status:
            return
        if 'All_Osc_Runner' in self.status:
            return

        shape = (len(self._hs), len(self._fs))
        if indices == 'Auto':
            indices = [i for i in numpy.ndindex(shape)]
            numpy.random.shuffle(indices)

        def run_all(indices):
            while indices:
                for i_th in range(n_threads):
                    th_filename = f'Osc{i_th}.mx3'
                    self.status  # update _running_threads
                    if not th_filename in [th.mx3_file for th
                                           in self._running_threads]:
                        if indices:
                            hi, fi = indices.pop()
                            self.Run_Osc(hi, fi, th_filename, gpu=gpu)
                if len(self._running_threads) > n_threads:
                    time.sleep(1)
                if self._stop:
                    indices = []
                    self._stop = False
        All_Osc_Runner = ThD.as_thread(run_all)
        All_Osc_Runner.info = (f'All_Osc_Runner GPU{gpu}')
        All_Osc_Runner.mx3_file = None
        All_Osc_Runner(indices)
        self._running_threads.append(All_Osc_Runner)

    def Run_Modes(self, hi, fi, file_name='Mode.mx3', gpu=None):
        if gpu is None:
            gpu = self._gpu
        if 'DC_Runner' in self.status:
            return
        if file_name in [th.mx3_file for th in self._running_threads]:
            return
        if not self._hi_fi_Done(hi, fi):
            self.Run_Osc(hi, fi, file_name)
            th = self._running_threads[-1]
            th.thread.join()
            return

        Sim_Tables = numpy.load(self._Sim_Tables_path, mmap_mode='r',
                                allow_pickle=True)
        t0 = Sim_Tables[hi, fi][0, 0]
        mode_data = {'RF_state': join(self._path, 'OscData',
                                      f'Osc_hi{hi}_fi{fi}.ovf'),
                     'h_dc': self._hs[hi],
                     't_0': t0,
                     'freq': self._fs[fi],
                     'DC_dir': tuple(self._DC_dir),
                     'RF_dir': tuple(self._RF_dir)
                     }
        _mx3_file = self._create_mx3_file(file_name=file_name,
                                          template=self._mx3_modes,
                                          data=mode_data)
        run_mumax3(_mx3_file, kernel_path=self.kernel_path, gpu=gpu)
        if file_name == 'Mode.mx3':
            self._last_mode = [hi, fi]

    def Run_StaticField(self, hi, field_name,
                        file_name='StaticField.mx3', gpu=None):
        if gpu is None:
            gpu = self._gpu
        if 'DC_Runner' in self.status:
            return
        if file_name in [th.mx3_file for th in self._running_threads]:
            return

        SF_data = {'DC_state': join(self._path,
                                    f'DC_sweep.out/Eq_hi{hi}.ovf'),
                   'h_dc': self._hs[hi],
                   'DC_dir': tuple(self._DC_dir),
                   'field_name': field_name
                   }
        _mx3_file = self._create_mx3_file(file_name=file_name,
                                          template=_mx3_static_field,
                                          data=SF_data)
        run_mumax3(_mx3_file, kernel_path=self.kernel_path, gpu=gpu)

    def Run_DynamicField(self, hi, fi, field_name,
                         file_name='DynamicField.mx3', gpu=None):
        if gpu is None:
            gpu = self._gpu
        if 'DC_Runner' in self.status:
            return
        if file_name in [th.mx3_file for th in self._running_threads]:
            return
        if not self._hi_fi_Done(hi, fi):
            self.Run_Osc(hi, fi, 'Mode.mx3')
            th = self._running_threads[-1]
            th.thread.join()

        if not self._last_mode == [hi, fi]:
            self.Run_Modes(hi, fi)

        Sim_Tables = numpy.load(self._Sim_Tables_path, mmap_mode='r',
                                allow_pickle=True)
        t0 = Sim_Tables[hi, fi][0, 0]
        DF_data = {'path': self._path,
                   'h_dc': self._hs[hi],
                   't_0': t0,
                   'hi': hi,
                   'fi': fi,
                   'freq': self._fs[fi],
                   'DC_dir': tuple(self._DC_dir),
                   'RF_dir': tuple(self._RF_dir),
                   'field_name': field_name
                   }
        _mx3_file = self._create_mx3_file(file_name=file_name,
                                          template=_mx3_dynamic_field,
                                          data=DF_data)
        run_mumax3(_mx3_file, kernel_path=self.kernel_path, gpu=gpu)

    def Stop(self):
        self._stop = True

    def PrintInfo(self):
        for k, v in self.Info.items():
            print(f'//****{k}****//')
            print(v)
            print(' ')

    def _hi_fi_Done(self, hi, fi):
        osc_file = join(self._OscData_path, f'Osc_hi{hi}_fi{fi}.ovf')
        return os.path.exists(osc_file)

    def _save_data_hi_fi(self, hi, fi, sim_dir):
        with self._Semaphore:
            table = numpy.loadtxt(join(sim_dir, 'table.txt'))
            osc_file = join(sim_dir, 'Osc_id.ovf')
            dest_file = join(self._OscData_path, f'Osc_hi{hi}_fi{fi}.ovf')
            os.rename(osc_file, dest_file)
            Sim_Tables = numpy.load(self._Sim_Tables_path,
                                    mmap_mode='r+',
                                    allow_pickle=True)
            Sim_Tables[hi, fi] = table[:, :7]
            del Sim_Tables

            if self._ExtraTableAdds > 0:
                Extra_Tables = numpy.load(self._Extra_Tables_path,
                                          mmap_mode='r+',
                                          allow_pickle=True)
                Extra_Tables[hi, fi] = table[:, 7:]
                del Extra_Tables

    @property
    def status(self):
        with self._Semaphore:
            self._running_threads = [rt for rt in self._running_threads
                                     if rt.thread.is_alive()]
            if self._running_threads:
                _status = [rt.info for rt in self._running_threads]
            else:
                _status = 'Idle'
        return _status

    @property
    def Info(self):
        data = {'mx3_base': self._mx3_base,
                'mx3_defs': self._mx3_defs,
                'mx3_eq': self._mx3_eq,
                'mx3_osc': self._mx3_osc,
                'mx3_modes': self._mx3_modes,
                'h': self._hs,
                'f': self._fs,
                'DC_dir': self._DC_dir,
                'RF_dir': self._RF_dir,
                'ExtraTableColumns': self._ExtraTableAdds}
        return data

    @property
    def hs(self):
        return self._hs

    @hs.setter
    def hs(self, crv):
        self._hs = numpy.atleast_1d(crv)

    @property
    def fs(self):
        return self._fs

    @fs.setter
    def fs(self, freqs):
        self._fs = numpy.atleast_1d(freqs)

    @property
    def DC_dir(self):
        return self._DC_dir

    @DC_dir.setter
    def DC_dir(self, dir_vector):
        self._DC_dir = dir_vector/numpy.linalg.norm(dir_vector)

    @property
    def RF_dir(self):
        return self._RF_dir

    @DC_dir.setter
    def RF_dir(self, dir_vector):
        self._RF_dir = dir_vector/numpy.linalg.norm(dir_vector)
