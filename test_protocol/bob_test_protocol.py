from sklearn.pipeline import Pipeline
from bob.pipelines.wrappers import SampleWrapper
from bob.bio.face.embeddings.pytorch import PyTorchModel
from bob.bio.base.algorithm import Distance
from bob.bio.base.pipelines import PipelineSimple, execute_pipeline_simple
import bob.bio.base.score
import bob.measure
from shutil import rmtree

import torch
import os

# Local client
from dask.distributed import Client
n_workers = 10


class MyRunnableModel(PyTorchModel):
    """
    MyRunnableModel

    Receives a Pytorch module instance and makes it runnable as an estimator.

    Parameters:
    ----------
        model:
          A loaded pytorch model (module instance)

        preprocessor:
          A function that will transform the data right before forward. The default transformation is `X/255`

        memory_demanding: pipeline parameter

        device:

    """

    def __init__(
        self,
        model,
        preprocessor=lambda x: x / 255,
        memory_demanding=False,
        device=None,
        **kwargs,
    ):
        super(MyRunnableModel, self).__init__(
            preprocessor=preprocessor,
            memory_demanding=memory_demanding,
            device=device,
            **kwargs,
        )

        self.model = model
        self.model.eval()
        self.is_loaded_by_function = False

    def _load_model(self):
        self.model = self._model_fn()
        self.model.eval()

    def __getstate__(self):
        if self.is_loaded_by_function:
            return super(MyRunnableModel, self).__getstate__()

# ON OUR SGE
def scale_to_sge(n_workers):
    queue = "q_1day"
    queue_resource_spec = "q_1day=TRUE"
    memory = "30GB"
    sge_log = "/idiap/temp/lluevano/DEBUG_TINYFACE_DASK/logs"
    from dask_jobqueue import SGECluster
    cluster = SGECluster(queue=queue, memory=memory, cores=1, processes=1,
              log_directory=sge_log,
              local_directory=sge_log,
              resource_spec=queue_resource_spec,
              project="scbiometrics",
              )
    cluster.scale_up(n_workers)
    return Client(cluster)  # start local workers as threads


#### SWITH THIS IF YOU WANT TO RUN LOCALLY OR IN OUR SGE GRID ###

# Local client
#client = Client(n_workers=n_workers)

# SGE client

def get_database(name):
    if name=="tinyface":
        from bob.bio.face.config.database import tinyface
        database = tinyface.database
    return database

def score_bob_model(model, db_name="tinyface", out_dir="/idiap/temp/lluevano/DEBUG_TINYFACE_DASK", groups=["dev",], epoch="last"):
    #Run scores pipeline

    #  FaceXZoo model has the backbone and head modules. We select the backbone for verification
    backbone = model
    bob_model = MyRunnableModel(backbone, preprocessor=lambda x: (x-127.5)/128.0, device=torch.device('cuda'))

    transformer = [('embedding', SampleWrapper(estimator=bob_model,
                                               fit_extra_arguments=(), input_attribute='data',
                                               output_attribute='data'))]

    transformer_pipeline = Pipeline(transformer)
    biometric_algorithm = Distance('cosine')
    simple_pipeline = PipelineSimple(transformer_pipeline, biometric_algorithm)

    database = get_database(db_name)

    #client = scale_to_sge(n_workers)
    #Compute only dev set for time constraints

    execute_pipeline_simple(
        pipeline=simple_pipeline,
        database=database,
        dask_client=None, # TODO: add dask client interface
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

    scores = []
    for group in groups:
        print(f"Measures for group {group}")
        fname = os.path.join(out_dir,f"scores-{group}.csv")
        negatives, positives = bob.bio.base.score.load.split(fname)
        rr_scores = bob.bio.base.score.load.cmc(fname)
        R1 = bob.measure.recognition_rate(rr_scores, rank=1) * 100
        EER = bob.measure.eer(negatives, positives, is_sorted=False, also_farfrr=False) * 100
        AUC = bob.measure.roc_auc_score(negatives, positives, npoints=2000, min_far=- 8, log_scale=False) * 100
        scores.append({'R1':R1, 'EER':EER, 'AUC': AUC})
        os.rename(fname,os.path.join(out_dir,f"scores-{group}-{epoch}.csv"))
        print(scores)
    return scores


