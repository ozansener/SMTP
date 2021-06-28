import click
import json
import gym
import socket
import ray

from dfo_learner import DFOLearner

@click.command()
@click.option('--param_file', default='params.json', help='JSON file for exp parameters')
def train_dfo(param_file):
    with open(param_file) as json_params:
        params = json.load(json_params)
    
    exp_identifier = '|'.join('{}={}'.format(key,val) for (key,val) in params.items())
    exp_identifier = "".join(exp_identifier.split())
    params['exp_id'] = exp_identifier

    if 'LQR' in params['env_name']:
        env = LQRKakade()
    else:
        env = gym.make(params['env_name'])
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # set policy parameters. Possible filters: 'MeanStdFilter' for v2, 'NoFilter' for v1.
    policy_params={'type':'linear',
                   'ob_filter': "MeanStdFilter",
                   'ob_dim':obs_dim,
                   'ac_dim':act_dim}
    
    params["policy_params"] = policy_params

    local_ip = socket.gethostbyname(socket.gethostname())
    ray.init(redis_address= local_ip + ':6389')

    model = DFOLearner(params)
    model.train(params['n_iter'])


if __name__ == '__main__':
    train_dfo()

