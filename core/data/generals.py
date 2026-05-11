# ===== camera configurations =====
EGO_CAMERA_MAP = {
    # 'dataset name': 'ego camera key name',
    'AgiBotAlpha_covt2lerobot_0818': 'observation.images.cam_2',
    'AgiBotBeta_covt2lerobot_0818': 'observation.images.cam_2',
    'austin_buds_dataset_converted_externally_to_rlds': 'observation.images.cam1',
    'austin_sailor_dataset_converted_externally_to_rlds': 'observation.images.cam1',
    'austin_sirius_dataset_converted_externally_to_rlds': 'observation.images.cam1',
    'stanford_hydra_dataset_converted_externally_to_rlds': 'observation.images.cam1',
    'iamlab_cmu_pickup_insert_converted_externally_to_rlds': 'observation.images.cam1',
    'nyu_franka_play_dataset_converted_externally_to_rlds': 'observation.images.cam',
    'robo_set_new': 'observation.images.cam3',
    'fmb': 'observation.images.cam3',
    'berkeley_autolab_ur5': 'observation.images.cam1',
    'berkeley_fanuc_manipulation': 'observation.images.cam1',
    'qut_dexterous_manpulation': 'observation.images.cam1',
    'taco_play': 'observation.images.cam1',
    'jaco_play': 'observation.images.cam1',
    'utaustin_mutex': 'observation.images.cam1',
    'plex_robosuite': 'observation.images.cam1',
}
EXO_CAMERA_MAP = {
    # 'dataset name': 'third camera key name',
    'bridge_orig_lerobot': 'observation.images.image_0',
    'nyu_franka_play_dataset_converted_externally_to_rlds': 'observation.images.cam1',
}



# ===== dataset configurations =====
from enum import IntFlag, auto
class DatasetStatus(IntFlag):
    INVALID         = 0        # 0 as unusual case
    VANILLA         = 1 << 0   # 000001
    WITH_PLAN       = 1 << 1   # 000010
    WITH_SUB_TASK   = 1 << 2   # 000100
    WITH_COT        = 1 << 3   # 001000
    WITH_BBOX       = 1 << 4   # 010000



# ===== generate dataset lists =====
# deprecated