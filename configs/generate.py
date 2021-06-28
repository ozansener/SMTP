import json

data = json.load(open('swimmer.json'))
print(data)

valid_delta = [0.001, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0]
valid_workers = [1, 2, 4]
valid_beta = [0.1, 0.5, 0.9, 0.99]

i = 0
for step in valid_delta:
    for worker in valid_workers:
        for beta in valid_beta:
            data['momentum'] = True
            data['step_size'] = step 
            data['delta_std'] = 'NA'
            data['num_workers'] = worker 
            data['num_rollouts'] = worker 
            data['beta'] = beta 
            #print(i)
            i+=1
            #with open('swimmer_momentum_{}.json'.format(i), 'w') as f:
            #    f.write(json.dumps(data, indent=4, sort_keys=True))

data['momentum'] = False

valid_delta = [0.01, 0.02, 0.04, 0.08, 0.16]
valid_workers = [1, 2, 4]

i = 0
for step in valid_delta:
    for delta in valid_delta:
        for worker in valid_workers:
            if delta > step:
                continue
            data['momentum'] = False
            data['step_size'] = step 
            data['delta_std'] = delta
            data['num_workers'] = worker 
            data['num_rollouts'] = worker 
            data['beta'] = 'NA' 
            print(i)
            i+=1
            with open('swimmer_{}.json'.format(i), 'w') as f:
                f.write(json.dumps(data, indent=4, sort_keys=True))

