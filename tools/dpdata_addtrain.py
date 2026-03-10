import dpdata
import argparse

parser = argparse.ArgumentParser("dpdata merger")

parser.add_argument("origin_dpdata", help="original training dpdata, usually in format deepmd/npy/mixed")
parser.add_argument("added_dpdata", help="dpdata added to training data, usually in format deepmd/npy")
parser.add_argument("-o", "--output", help="dirname of output training data, usually in format deepmd/npy/mixed", default="sampled_dpdata")
parser.add_argument("--type_ori", help="data type for original dpdata", default="deepmd/npy/mixed")
parser.add_argument("--type_add", help="data type for added dpdata", default="deepmd/npy")
#parser.add_argument("--type_out", help="data type for output training dpdata", default="deepmd/npy/mixed")

args = parser.parse_args()

origin_dpdata = dpdata.MultiSystems.from_file(args.origin_dpdata, fmt=args.type_ori)
print(f"origin dpdata: {origin_dpdata}")
added_dpdata = dpdata.MultiSystems.from_file(args.added_dpdata, fmt=args.type_add)
print(f"added dpdata: {added_dpdata}")
output_dpdata = origin_dpdata + added_dpdata
print(f"output dpdata: {output_dpdata}")

output_dpdata.to_deepmd_npy_mixed(args.output)

print("Done!")
