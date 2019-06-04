from django.conf import settings
import sys

projectPath = "/".join(settings.BASE_DIR.split('/')[:-1])
sys.path.append(projectPath)
# print(chatbotPath)

from engines.BiLSTM_CRFs import BiLSTM_CRFs
from engines.Configer import Configer
from engines.DataManager import DataManager
from engines.utils import get_logger

configs = Configer(config_file=projectPath + "/system.config")
configs.datasets_fold = (projectPath + "/" + configs.datasets_fold)
configs.vocabs_dir = (projectPath + "/" + configs.vocabs_dir)
configs.token_emb_dir = (projectPath + "/" + configs.token_emb_dir)
configs.checkpoints_dir = (projectPath + "/" + configs.checkpoints_dir)
configs.log_dir = (projectPath + "/" + configs.log_dir)

logger = get_logger(configs.log_dir)
dataManager = DataManager(configs, logger)
model = BiLSTM_CRFs(configs, logger, dataManager)
model.soft_load()
model.predict_single(" ") #warm start
color_list = ['19AC8D', 'FFF000', '6892B9', '59AC6F', 'E87F78', '335ABF', 'DEC778',
              'B581E4', 'B93326', 'B1B99C']
color_list = color_list * 5

if configs.label_level ==2:
    color_dict = {typ: color_list[i] for i, typ in enumerate(configs.suffix)}
