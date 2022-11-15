import os
from shutil import rmtree

import torch
from sklearn.pipeline import Pipeline
from bob.bio.face.embeddings.pytorch import PyTorchModel
from bob.bio.base.algorithm import Distance
from bob.bio.base.pipelines import PipelineSimple, execute_pipeline_simple
import bob.bio.base.score
import bob.measure
from bob.pipelines import wrap

class MyModel(PyTorchModel):

    def __init__(
        self,
        preprocessor=lambda x: (x - 127.5) / 128.0,
        memory_demanding=False,
        device=None,
        **kwargs,
    ):
        super(MyModel, self).__init__(
            None,
            None,
            memory_demanding=memory_demanding,
            preprocessor=preprocessor,
            device=device,
            **kwargs,
        )

    def _load_model(self): # used by PyTorchModel
        self.model = self.module
        self.model.eval()
        self.place_model_on_device()

    def pass_module(self, module): # set the Pytorch Module model
        self.module = module

def get_database(name):
    # returns bob database instance
    if name == "tinyface":
        from bob.bio.face.config.database import tinyface
        database = tinyface.database
    elif name == "survface":
        from bob.bio.face.config.database import survface
        database = survface.database
    else:
        raise ValueError(f"Dataset {name} not available")
    return database

def measure_bob_scores(fname):
    negatives, positives = bob.bio.base.score.load.split(fname)
    rr_scores = bob.bio.base.score.load.cmc(fname)
    R1 = bob.measure.recognition_rate(rr_scores, rank=1) * 100
    EER = bob.measure.eer(negatives, positives, is_sorted=False, also_farfrr=False) * 100
    AUC = bob.measure.roc_auc_score(negatives, positives, npoints=2000, min_far=- 8, log_scale=False) * 100
    # threshold = 0.001
    # bob.measure.farfrr(negatives, positives, threshold)
    score_dict = {'R1': R1, 'EER': EER, 'AUC': AUC}
    return score_dict
def score_bob_model(model, db_name="tinyface", out_dir="/idiap/temp/lluevano/DEBUG_TINYFACE_DASK", groups=["dev",], epoch="last", device='cuda', dask_client=None):
    #Run scores pipeline

    #  FaceXZoo model has the backbone and head modules. We select the backbone for verification
    backbone = model
    bob_model = MyModel(device=torch.device("cpu"))
    bob_model.pass_module(backbone)

    transformer = [('embedding', wrap(["sample"], bob_model))]

    transformer_pipeline = Pipeline(transformer)
    biometric_algorithm = Distance('cosine')
    simple_pipeline = PipelineSimple(transformer_pipeline, biometric_algorithm)

    database = get_database(db_name)

    execute_pipeline_simple(
        pipeline=simple_pipeline,
        database=database,
        dask_client=dask_client,
        groups=groups,
        output=out_dir,
        write_metadata_scores=True,
        checkpoint=True,
        dask_n_partitions=None,
        dask_partition_size=None,
        dask_n_workers=None,
        checkpoint_dir=None,
        force=False,
    )
    print("Scores written")

    print("Cleaning up...")

    rmtree(os.path.join(out_dir, "biometric_references"))
    rmtree(os.path.join(out_dir, "embedding"))
    rmtree(os.path.join(out_dir, "tmp"))
    #rmtree(os.path.join(out_dir, "cropper"))

    scores = []
    for group in groups:
        fname_orig = os.path.join(out_dir, f"scores-{group}.csv")
        fname = os.path.join(out_dir, f"scores-{group}-{epoch}.csv")
        os.rename(fname_orig, fname)
        print(f"Measures for group {group}")
        score_dict = measure_bob_scores(fname)
        scores.append(score_dict)
        print(scores)
    return scores


