import numpy as np
import os
from mpi4py import MPI
from common import info, info_split, info_blue, info_cyan, info_on_red, info_warning
import h5py
import glob
from xml.etree import cElementTree as ET

mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()

class Params():
    def __init__(self, *args):
        self.dolfin_params = dict()

        if len(args) > 0:
            input_file = args[0]
            with open(input_file, "r") as infile:
                for el in infile.read().split("\n"):
                    if "=" in el:
                        key, val = el.split("=")
                        if val in ["true", "TRUE"]:
                            val = "True"
                        elif val in ["false", "FALSE"]:
                            val = "False"
                        try:
                            self.dolfin_params[key] = eval(val)
                        except:
                            self.dolfin_params[key] = val


    def __getitem__(self, key):
        if key in self.dolfin_params:
            return self.dolfin_params[key]
        else:
            exit("No such parameter: {}".format(key))
            #return None

    def __setitem__(self, key, val):
        self.dolfin_params[key] = val

    def __str__(self):
        string = ""
        for key, val in self.dolfin_params.items():
            string += "{}={}\n".format(key, val)
        return string

    def save(self, filename):
        if mpi_rank == 0:
            with open(filename, "w") as ofile:
                ofile.write(self.__str__())
    
def get_next_subfolder(folder):
    if mpi_rank == 0:
        i = 0
        while os.path.exists(os.path.join(folder, "{}".format(i))):
            i += 1
    else:
        i = None
    i = mpi_comm.bcast(i, root=0)
    return os.path.join(folder, "{}".format(i))

def makedirs_safe(folder):
    """ Make directory in a safe way. """
    if mpi_rank == 0 and not os.path.exists(folder):
        os.makedirs(folder)

def remove_safe(path):
    """ Remove file in a safe way. """
    if mpi_rank == 0 and os.path.exists(path):
        os.remove(path)

def numpy_to_dolfin_file(nodes, elements, filename):
    """ Convert nodes and elements to a (legacy) dolfin mesh file. """

    dim = elements.shape[1]-1
    cell_type = "triangle"
    if dim == 3:
        cell_type = "tetrahedron"

    if mpi_rank == 0:
        with h5py.File(filename, "w") as h5f:
            cell_indices = h5f.create_dataset(
                "mesh/cell_indices", data=np.arange(len(elements)),
                dtype='int64')
            topology = h5f.create_dataset(
                "mesh/topology", data=elements, dtype='int64')
            coordinates = h5f.create_dataset(
                "mesh/coordinates", data=nodes, dtype='float64')
            topology.attrs["celltype"] = np.string_(cell_type)
            topology.attrs["partition"] = np.array([0], dtype='uint64')

    mpi_comm.Barrier()


def parse_xdmf(xml_file, get_mesh_address=False):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    dsets = []

    geometry_found = not get_mesh_address
    topology_found = not get_mesh_address

    for i, step in enumerate(root[0]):
        if step.tag == "Grid":
            timestamp = None
            dset_address = None
            for prop in step:
                if not topology_found and prop.tag == "Topology":
                    topology_address = prop[0].text.split(":")
                    topology_address[0] = os.path.join(
                        os.path.dirname(xml_file), topology_address[0])
                    topology_found = True
                elif not geometry_found and prop.tag == "Geometry":
                    geometry_address = prop[0].text.split(":")
                    geometry_address[0] = os.path.join(
                        os.path.dirname(xml_file), geometry_address[0])
                    geometry_found = True
                elif prop.tag == "Grid":
                    for sprop in prop:
                        if sprop.tag == "Time":
                            timestamp = float(sprop.attrib["Value"])

                        elif sprop.tag == "Attribute":
                            dset_name = sprop.attrib["Name"]
                            #print(dset_name)
                            dset_address = sprop[0].text.split(":")
                            dset_address[0] = os.path.join(
                                os.path.dirname(xml_file), dset_address[0])
                            #dsets.append((timestamp, dset_name, dset_address))
                            dsets.append((timestamp, dset_address))

    if get_mesh_address and topology_found and geometry_found:
        return (dsets, topology_address, geometry_address)
    return dsets


