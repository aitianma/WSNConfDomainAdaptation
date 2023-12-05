import argparse
from experiment import test,experimentWithDiffSampleNumber
from MatSampler import MatSampler

parser = argparse.ArgumentParser(description='[TSM] Teacher Student Model')

#parameter to set sampling algorith enabled or not
parser.add_argument('--sampling', type=str, required=False, default=True,help='sampling algorithm enabled or not, options: [True,False]')

#parameter to set which experiment to run
parser.add_argument('--exp', type=str, required=True, default="diff",help='run sampling expperiment or the experiment with different number of physical samples, options: ["sampling","diff","draw"]')



args = parser.parse_args()

if args.exp=="diff":
    experimentWithDiffSampleNumber()

elif args.exp=="sampling":
    print(args.sampling)

    if args.sampling=="True":
        sampling_enabled = True
    else:
        sampling_enabled = False

    test(sampling_enabled=sampling_enabled)

elif args.exp=="mahalanobis":

    ms = MatSampler(simulator="tossim")
    sc= ms.getMahalanobisInEachNC()
