
import os
import random

import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch import Tensor

from .mcts_args import MctsArgs

from .intf_py_communicator import PyCommunicator

from .othello_nnet import OthelloNNet as onnet

from numpy.typing import NDArray
from numpy import float32
import logging
from .train_example import TrainExample

log = logging.getLogger(__name__)


class AverageMeter:
    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def __repr__(self):
        return f"{self.avg:.2e}"

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class NNetWrapper:
    def __init__(self, pc: PyCommunicator, args: MctsArgs):
        self.args = args
        self.nnet = onnet(pc, self.args)
        self.board_x = pc.size_x()
        self.board_y = pc.size_y()

        self.action_size = pc.move_len()
        self.batch_size = pc.batch_size()

        if self.args.cuda:
            self.nnet.cuda()

    def train(self, examples: list[TrainExample]):
        args = self.args
        # いつもpylanceはこれに文句言うけど言われた通り直すとエラーになる
        optimizer = optim.Adam(self.nnet.parameters())  # type: ignore

        for epoch in range(args.epochs):
            print("EPOCH ::: " + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / self.batch_size)
            random.shuffle(examples)

            t = tqdm(range(batch_count), desc="Training Net")
            for i in t:
                if args.random_choice:
                    sample_ids = np.random.randint(  # type: ignore
                        len(examples), size=self.batch_size)
                    boards, target_pis, target_vs = map(
                        list,
                        zip(
                            *(
                                ((examples[id].canonical_board, examples[id].pi, examples[id].v)  # type: ignore
                                 for id in sample_ids)
                            )
                        ),
                    )
                else:
                    # 相当意味のわかりにくいコード。アンパックで引数をn個にしてzipにぶちこむスクリプト言語特有のやりかた
                    # zipが可変引数だからn個ぶち込んでしまえる
                    # 3個のlistを返すiterableになるので、分解できる（これもスクリプト言語特有のやり方)
                    boards, target_pis, target_vs = map(
                        list,
                        zip(
                            *(
                                (example.canonical_board, example.pi, example.v)
                                for example in examples[i*self.batch_size:(i+1)*self.batch_size]
                            )
                        ),
                    )

                boards = np.array(boards).astype(np.float32)
                target_pis = np.array(target_pis).astype(np.float32)
                target_vs = np.array(target_vs).astype(np.float32)

                # なんでfloat64なのかさっぱりわからない。遅いでしょ
                # boards = torch.Tensor(np.array(boards).astype(np.float64))
                boards = torch.from_numpy(  # type: ignore
                    boards).contiguous().cuda()
                target_pis = torch.from_numpy(  # type: ignore
                    target_pis).contiguous().cuda()
                # target_vs = torch.Tensor(np.array(vs).astype(np.float64))
                target_vs = torch.from_numpy(  # type: ignore
                    target_vs).contiguous().cuda()

                out_pi, out_v = self.nnet(boards)

                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)

                total_loss = l_pi + l_v

                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses,  # type: ignore
                              Loss_v=v_losses)

                optimizer.zero_grad()
                total_loss.backward()  # type: ignore
                optimizer.step()  # type: ignore

    def predict(self, board: NDArray[float32]) -> tuple[NDArray[float32], NDArray[float32]]:
        # start = time.time()

        board: Tensor = torch.Tensor(board)
        if self.args.cuda:
            board = board.contiguous().cuda()

        board = board.view(self.batch_size, self.board_x, self.board_y)

        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        assert isinstance(pi, Tensor)
        assert isinstance(v, Tensor)

        r1 = torch.exp(pi).data.cpu().numpy()  # type: ignore
        r2 = v.data.cpu().numpy()  # type: ignore

        return (r1, r2)  # type: ignore

    def loss_pi(self, targets: Tensor, outputs: Tensor) -> Tensor:
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets: Tensor, outputs: Tensor) -> Tensor:
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(
        self, folder: str = "checkpoint", filename: str = "checkpoint.pth.tar"
    ):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(
                    folder
                )
            )
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save(  # type: ignore
            {
                "state_dict": self.nnet.state_dict(),
            },
            filepath,
        )

    def load_checkpoint(
        self, folder: str = "checkpoint", filename: str = "checkpoint.pth.tar"
    ):
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise IOError("No model in path {}".format(filepath))
        map_location = None if self.args.cuda else "cpu"
        checkpoint = torch.load(  # type: ignore
            filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint["state_dict"])
