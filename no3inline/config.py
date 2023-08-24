#################################################################
# interesting values of N
# 1. 10-16 : can we find solution? we can verify as all are known
# 2. 17-18 : unpublished solutions may exist
# 3. 19-46 : not all solutions are known
# 4. 48-50-52 : a single solution is known
# 5. 47-49-51-52< : no solution is known
#################################################################    

HYPERPARAMETERS = {
    'RUN_NAME': 'test_N=14',
    'PROJECT': 'testing',
    'N': 14,
    'MODEL': 'convnet', # 'convnet' or 'resnet'
    'LEARNING_RATE': 0.0005,
    'N_ROLLOUTS': 100,
    'N_EPOCHS': 1000,
    'N_ITER': 100,
    'TOP_K_PERCENT': 0.05,
    'EPSILON': 0.2, # probability of random action
    'REWARD_TYPE': 'laststate', # 'summed' or 'laststate',
    'DEDUPLICATION': True
}
