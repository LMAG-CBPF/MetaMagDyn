# -*- coding: utf-8 -*-

##########################################################
# File for extracting the data from an OOMMF OVF 2.0 file
##########################################################

import numpy

"""
Author: Diego González Chávez
email : diegogch@cbpf.br / diego.gonzalez.chavez@gmail.com

Loosely inspired by the oommfdecode.py file by Mark Mascaro
"""


class OOMMF_Vector_Field(object):
    def __init__(self):
        pass

    def load(self, fileName):
        data_file = open(fileName, 'rb')
        a = 'random string'

        #################################################
        # skim over the file header, and find the
        # following:
        #     xnodes, ynodes, znodes
        #     dx, dy, dz
        #     Total simulation time
        #################################################

        while a.find('Begin: Data') == -1:
            a = data_file.readline()[:-1].decode('ascii')
            if a.find('Total simulation time') != -1:
                t = float(a.split()[5])

            elif a.find('xnodes') != -1:
                xnodes = int(a.split()[2])
            elif a.find('ynodes') != -1:
                ynodes = int(a.split()[2])
            elif a.find('znodes') != -1:
                znodes = int(a.split()[2])

            elif a.find('xstepsize') != -1:
                xbase = float(a.split()[2])
            elif a.find('ystepsize') != -1:
                ybase = float(a.split()[2])
            elif a.find('zstepsize') != -1:
                zbase = float(a.split()[2])

            elif a.find('valuedim') != -1:
                valuedim = int(a.split()[2])

        self.t = t
        self.xnodes = xnodes
        self.ynodes = ynodes
        self.znodes = znodes
        self.dx = xbase
        self.dy = ybase
        self.dz = zbase
        self.dim = valuedim

        # Get the record size of binary data
        dataSize = int(a.split()[-1])
        # Get the OMF_CONTROL_NUMBER in big endian format
        dataType = '>f' + str(dataSize)
        t1 = numpy.fromfile(data_file, dtype=dataType, count=1)
        # If encoding is incorrect, change it to little endian
        if str(t1).find('1234567') == -1:
            dataType = '<f' + str(dataSize)

        # get all the data
        self.data = numpy.fromfile(data_file, dtype=dataType,
                                   count=xnodes*ynodes*znodes*valuedim)
        data_file.close()

        self.data = self.data.reshape((valuedim, xnodes, ynodes, znodes),
                                      order='F')
        self.data = numpy.rollaxis(self.data, 0, 4)
        if valuedim == 3:
            self.vx = self.data[:, :, :, 0]
            self.vy = self.data[:, :, :, 1]
            self.vz = self.data[:, :, :, 2]


def load_OOMMF_VectorField(fileName):
    V = OOMMF_Vector_Field()
    V.load(fileName)

    return V
