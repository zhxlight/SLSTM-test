import _pickle
import os

import torch
from fastNLP import Adam
from fastNLP import Trainer, Tester, CrossEntropyLoss, AccuracyMetric

from args import get_args
from model.BertModel import TextClassificationModel, BertSLSTMTextClassificationModel
from preprocess import load_dataset

arg = get_args()
for k in arg.__dict__.keys():
    print(k, arg.__dict__[k])

save_dir = os.path.join("./save", "bert")
os.environ['CUDA_VISIBLE_DEVICES'] = arg.gpu

if os.path.exists(f'./cache/{arg.dataset}_train_dataset.pkl'):
    train_dataset = _pickle.load(open(f'./cache/{arg.dataset}_train_dataset.pkl', 'rb'))
    dev_dataset = _pickle.load(open(f'./cache/{arg.dataset}_dev_dataset.pkl', 'rb'))
    test_dataset = _pickle.load(open(f'./cache/{arg.dataset}_test_dataset.pkl', 'rb'))
else:
    train_dataset = load_dataset(data_dir=arg.data_dir, data_path=arg.dataset + '_trn')
    dev_dataset = load_dataset(data_dir=arg.data_dir,data_path=arg.dataset + '_dev')
    test_dataset = load_dataset(data_dir=arg.data_dir,data_path=arg.dataset + '_tst')

    # dataset = combine_data_set(train_dataset, dev_dataset)

with open(f'./cache/{arg.dataset}_train_dataset.pkl', 'wb') as f:
    _pickle.dump(train_dataset, f)
with open(f'./cache/{arg.dataset}_dev_dataset.pkl', 'wb') as f:
    _pickle.dump(dev_dataset, f)
with open(f'./cache/{arg.dataset}_test_dataset.pkl', 'wb') as f:
    _pickle.dump(test_dataset, f)

model = BertSLSTMTextClassificationModel()

trainer = Trainer(
    train_data=train_dataset,
    model=model,
    loss=CrossEntropyLoss(pred='predict', target='label'),
    metrics=AccuracyMetric(),
    n_epochs=20,
    batch_size=arg.batch_size,
    print_every=1,
    validate_every=-1,
    dev_data=dev_dataset,
    use_cuda=True,
    save_path=save_dir,
    optimizer=Adam(1e-3, weight_decay=0),
    check_code_level=-1,
    metric_key='acc',
    # sampler=default,
    use_tqdm=True,
)

results = trainer.train(load_best_model=True)
print(results)

torch.save(model, os.path.join(save_dir,"best_model.pkl"))


tester = Tester(
    data=test_dataset,
    model=model,
    metrics=AccuracyMetric(),
    batch_size=arg.batch_size,
    use_cuda=False,
)

eval_results = tester.test()
print(eval_results)
