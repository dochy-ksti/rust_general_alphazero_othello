import logging

from .arena import Arena

from .nnet import NNetWrapper

from .mcts_args import MctsArgs

import sys

from .intf_py_communicator import PyCommunicator


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

log = logging.getLogger(__name__)


def main():
    args = MctsArgs()
    try:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
    except IndexError:
        log.info('you need two files to compare')
        return

    pc = PyCommunicator(args.is_release)
    net1 = NNetWrapper(pc, args)
    net2 = NNetWrapper(pc, args)

    log.info('loading files...')
    if file1 != "none":
        net1.load_checkpoint(folder="target", filename=file1)
    if file2 != "none":
        net2.load_checkpoint(folder="target", filename=file2)

    arena = Arena()

    p1wins, p2wins, draws = arena.play_games(pc, net1, net2)

    log.info("P1/P2 WINS : %d / %d ; DRAWS : %d" %
             (p1wins, p2wins, draws))


if __name__ == "__main__":
    main()
