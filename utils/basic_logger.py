import sys


class BasicLogger:
    def __init__(self, criterion, metrics, save_path, epochs):
        self.criterion_name = criterion.name
        self.metrics_names = [m.name for m in metrics]
        self.save_path = save_path
        self.epochs = epochs

        self.mode = None
        self.current_epoch = 0
        self.current_step = 0
        self.dataset_length = 0
        self.batch_size = 0

        self.loss = 0
        self.metrics_results = [0 for _ in metrics]

    def progress_message(self):
        message = f'\r{self.mode} ({self.current_step}/{self.dataset_length}) -> '
        rounded_loss = round(float(self.loss) / (self.current_step + 1), 3)
        message += f'{self.criterion_name} loss = {rounded_loss}, '
        for i, met_name in enumerate(self.metrics_names):
            rounded_metric = round(float(self.metrics_results[i]) / ((self.current_step + 1) * self.batch_size), 3)
            message += f'{met_name} = {rounded_metric}    '
        message += '           '
        sys.stdout.write(message)

    def save_results(self):
        with open(self.save_path + '_results.txt', 'a') as f:
            message = f'{self.mode}: epoch {self.current_epoch + 1}/{self.epochs} -> '
            loss = float(self.loss) / (self.dataset_length * self.batch_size)
            message += f'{self.criterion_name} loss = {loss}, '
            for i, met_name in enumerate(self.metrics_names):
                message += f'{met_name} = {float(self.metrics_results[i]) / (self.dataset_length * self.batch_size)} '
            message += '\n'
            f.write(message)

    def new_epoch(self, current_epoch):
        self.current_epoch = current_epoch

    def train(self, dataset):
        self.mode = 'Train'
        self.current_step = 0
        self.loss = 0
        self.metrics_results = [0 for _ in self.metrics_names]
        self.dataset_length = len(dataset)
        self.batch_size = dataset.batch_size

    def eval(self, dataset):
        self.mode = 'Test'
        self.current_step = 0
        self.loss = 0
        self.metrics_results = [0 for _ in self.metrics_names]
        self.dataset_length = len(dataset)
        self.batch_size = dataset.batch_size

    def new_step(self, loss, metrics_results):
        self.current_step += 1
        self.loss += loss.item()
        for i, m in enumerate(metrics_results):
            self.metrics_results[i] += m


OBJECT = BasicLogger
