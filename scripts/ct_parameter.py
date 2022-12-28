import argparse
from argparse import RawTextHelpFormatter
import sys
import os
import subprocess

HEADER_ = """
#ifndef CT_PARAMETER_H
#define CT_PARAMETER_H

#include "patch_base.cuh"

namespace CT
{
"""

FOOTER_ = """
} // namespace CT

#endif // CT_PARAMETER_H
"""

BUILD_DIR_ = os.path.abspath("/export/home/cucui/CLionProjects/GPUTensorProductSmoothers") #  replaced by CMake
BUILD_INC_ = os.path.join(BUILD_DIR_,'include')
DEFAULT_OUTFILE_=os.path.join(BUILD_INC_,'ct_parameter.h')

def parse_args():
    parser = argparse.ArgumentParser(
        description="""Create compile-time parameters.""",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument('-O', '--output',
                        help="output file of the compile-time parameters",
                        type=argparse.FileType('w'),
                        default=DEFAULT_OUTFILE_
    )
    parser.add_argument('-DIM','--dimension',
                        default=2,
                        type=int,
                        choices=[2,3],
                        help="spatial dimension of the domain"
    )
    parser.add_argument('-DEG','--fe-degree',
                        default=2,
                        type=int,
                        choices=range(0,32),
                        help="spatial dimension of the domain"
    )
    parser.add_argument('-DLY','--dof-layout',
                        default='Q',
                        type=str,
                        choices=['DGQ','Q','RT'],
                        help="dof layout of finite element method"
    )
    parser.add_argument('-LOG','--log-directory',
                        default=BUILD_DIR_,
                        help="directory of the log file"
    )
    parser.add_argument('-K','--kernel_type',
                        default='FUSED',
                        choices=['GLOBAL','SEPERATE','FUSED'],
                        help="variant of the Schwarz smoother kernel"
    )
    parser.add_argument('-G','--granularity',
                        default='none',
                        choices=['none','user_define','multiple'],
                        help="thread-block granularity scheme"   
    )                                     
    parser.add_argument('-VNUM','--vcycle_number',
                        default='double',
                        type=str,
                        choices=['double','float'],
                        help="number type for the multigrid v-cycle"
    )
    args = parser.parse_args()
    assert os.path.isdir(args.log_directory),"Invalid directory path: {}".format(args.log_directory)
    fpath = args.output.name
    assert os.path.isdir(os.path.dirname(fpath)), "Invalid output director: {}".format(os.path.dirname(fpath))
    return args

def rawstr(string):
    return (r'"' + string + r'"')

class Parameter:
    """Class containing the compile-time parameters."""

    def __init__(self):
        """generating default and parsed parameters"""
        options = parse_args()
        dim = options.dimension
        deg = options.fe_degree
        self.dimension = ('constexpr unsigned int', dim)
        self.fe_degree = ('constexpr unsigned int', deg)
        self.dof_layout = ('constexpr auto', 'PSMF::DoFLayout::' + str(options.dof_layout))
        self.vcycle_number = ('using', str(options.vcycle_number))
        self.log_dir = ('const std::string', rawstr(options.log_directory))
        self.smoother_prm(options.kernel_type)
        self.granularity_prm(options.granularity)

    def smoother_prm(self,variant):
        """translate the variant into parameters"""
        if variant.startswith('F'):
            self.kernel_type = ('constexpr auto', r'PSMF::SmootherVariant::FUSED')
        elif variant.startswith('G'):
            self.kernel_type = ('constexpr auto', r'PSMF::SmootherVariant::GLOBAL')
        elif variant.startswith('S'):
            self.kernel_type = ('constexpr auto', r'PSMF::SmootherVariant::SEPERATE')

    def granularity_prm(self,variant):
        """translate the variant into parameters"""
        if variant.startswith('n'):
            self.granularity = ('constexpr auto', r'PSMF::GranularityScheme::none')
        elif variant.startswith('u'):
            self.granularity = ('constexpr auto', r'PSMF::GranularityScheme::user_define')
        elif variant.startswith('m'):
            self.granularity = ('constexpr auto', r'PSMF::GranularityScheme::multiple')         

def assignf(name_,type_,value_):
    """ formatted string defining a C-style assignment"""
    name = name_.upper() + '_'
    return "{} {} = {};".format(type_,name,value_)

def main():
    options = parse_args()
    ostream = options.output

    def oprint(*objects,**kwargs):
        """prints the output to the given output stream"""
        print(*objects,file=ostream,**kwargs)

    #: modify the parameters
    prm = Parameter()

    #: write header to output
    oprint(HEADER_,sep='\n')

    #: write variables to output
    def unpack_var(variables):
        """unpacks the variables into its name, type and value"""
        for name_ in variables:
            type_,value_ = variables[name_]
            yield name_,type_,value_
    variables = vars(prm)
    for name,vtype,value in unpack_var(variables):
        oprint(assignf(name,vtype,value),sep='\n')
    
    #: write footer to output
    oprint(FOOTER_,sep='\n')

    #: close the output stream
    if options.output:
        ostream.close()

if __name__ == '__main__':
    main()
