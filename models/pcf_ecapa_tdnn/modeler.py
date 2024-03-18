import torch

from models.ecapa_tdnn.loss import AAMsoftmax
from models.ecapa_tdnn.tools import MFCC
from models.pcf_ecapa_tdnn.model import PCFECAPATDNN
from models.modeler_interface import Modeler_interface


class Modeler(Modeler_interface):
    def __init__(self, speaker, device):
        super(Modeler, self).__init__(speaker, device)

        preprocess = MFCC()
        model = PCFECAPATDNN()
        criterion = AAMsoftmax(speaker)
        optimizer = torch.optim.Adam(model.parameters())
        self.__load__(preprocess, model, criterion, optimizer, device)
