import argparse
import logging
from fc_img import make_fc_img

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

parser = argparse.ArgumentParser()
parser.add_argument('--date', required=True)
parser.add_argument('--init_time', required=True)
parser.add_argument('--lead_time', required=True)
args = parser.parse_args()

make_fc_img(args.date, args.init_time, args.lead_time)