import sys
import yaml
sys.path.append('../../')

from modules.PDT import PDT
from modules.RRDBNet import RRDBNet
from modules.RRDBNet_prelu import RRDBNet as RRDBNet_prelu
class ModuleFactory:

    """Factory to produce head according to the head_conf.yaml

    Attributes:
        head_type(str): which head will be produce.
        head_param(dict): parsed params and it's value.
    """

    def __init__(self, module_type, module_conf_file):
        self.module_type = module_type
        with open(module_conf_file) as f:
            module_conf = yaml.load(f, Loader=yaml.FullLoader)
            self.module_param = module_conf[module_type]

    def get_module(self):
        if self.module_type == "PDT":
            in_channels = self.module_param['in_channels']
            pool_features = self.module_param['pool_features']
            use_se = self.module_param['use_se']
            use_bias = self.module_param['use_bias']
            use_cbam = self.module_param['use_cbam']
            mod = PDT(in_channels, pool_features, use_se, use_bias, use_cbam)
        elif self.module_type == "RRDBNet":
            in_nc = self.module_param['in_nc']
            out_nc = self.module_param['out_nc']
            nf = self.module_param['nf']
            nb = self.module_param['nb']
            gc = self.module_param['gc']
            mod = RRDBNet(in_nc, out_nc, nf, nb, gc=gc)
        elif self.module_type == "RRDBNet_prelu":
            in_nc = self.module_param['in_nc']
            out_nc = self.module_param['out_nc']
            nf = self.module_param['nf']
            nb = self.module_param['nb']
            gc = self.module_param['gc']
            mod = RRDBNet_prelu(in_nc, out_nc, nf, nb, gc=gc)
        else:
            pass

        return mod
