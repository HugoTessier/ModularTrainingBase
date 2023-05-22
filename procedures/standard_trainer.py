import torch
import sys
import os
import torch.nn.functional as F


class StandardTrainer:
    def __init__(self, epochs, optimizer, scheduler, metrics, criterion, device, debug, output_path, name):
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics
        self.criterion = criterion
        self.current_epoch = 0
        self.name = name
        self.output_path = output_path
        self.save_path = os.path.join(self.output_path, self.name)
        self.device = device
        self.debug = debug

    def save_state(self, model):
        with open(self.save_path + '_state.pt', 'wb') as f:
            torch.save({'current_epoch': self.current_epoch}, f)
        with open(self.save_path + '_model.pt', 'wb') as f:
            torch.save(model.state_dict(), f)
        with open(self.save_path + '_optimizer.pt', 'wb') as f:
            torch.save(self.optimizer.state_dict(), f)

    def load_state(self, model):
        with open(self.save_path + '_state.pt', 'rb') as f:
            self.current_epoch = torch.load(f)['current_epoch']
            self.scheduler.last_epoch = self.current_epoch
        with open(self.save_path + '_model.pt', 'rb') as f:
            model.load_state_dict(torch.load(f))
        with open(self.save_path + '_optimizer.pt', 'rb') as f:
            self.optimizer.load_state_dict(torch.load(f))

    def need_to_resume_previous_state(self):
        if os.path.exists(self.save_path + '_state.pt'):
            with open(self.save_path + '_state.pt', 'rb') as f:
                return self.current_epoch < torch.load(f)['current_epoch']
        else:
            return False

    def __call__(self, model, dataset):
        self.training_start_routine(model, dataset)
        while self.current_epoch < self.epochs:
            self.epoch_start_routine(model, dataset)
            self.train_one_epoch(model, dataset)
            self.epoch_mid_routine(model, dataset)
            self.test_one_epoch(model, dataset)
            self.epoch_end_routine(model, dataset)

    def epoch_start_routine(self, model, dataset):
        if self.need_to_resume_previous_state():
            self.load_state(model)
        else:
            self.save_state(model)
        print(f'Epoch {self.current_epoch + 1}/{self.epochs}')

    def epoch_mid_routine(self, model, dataset):
        print()

    def training_start_routine(self, model, dataset):
        print(self.name)
        if not os.path.isdir(self.output_path):
            try:
                os.mkdir(self.output_path)
            except OSError:
                print(f'Failed to create the folder {self.output_path}')
            else:
                print(f'Created folder {self.output_path}')

    def epoch_end_routine(self, model, dataset):
        print()
        self.scheduler.step()
        self.current_epoch += 1

    def progress_message(self, name, dataset, loss, metrics_results, current_step):
        message = f'\r{name} ({current_step + 1}/{len(dataset)}) -> '
        rounded_loss = round(float(loss) / (current_step + 1), 3)
        message += f'{self.criterion.name} loss = {rounded_loss}, '
        for k, m in enumerate(self.metrics):
            rounded_metric = round(float(metrics_results[k]) / ((current_step + 1) * dataset.batch_size), 3)
            message += f'{m.name} = {rounded_metric}\t'
        message += '           '
        sys.stdout.write(message)

    def train_one_epoch(self, model, dataset):
        results = [0 for _ in range(len(self.metrics))]
        global_loss = 0
        model.train()

        for i, (data, target) in enumerate(dataset['train']):
            if self.debug:
                if i != 0:
                    break

            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = model(data)
            loss = self.criterion(output, target.long())
            loss.backward()
            self.optimizer.step()

            for k, m in enumerate(self.metrics):
                results[k] += m(output, target)
            global_loss += loss.item()

            self.progress_message('Train', dataset['train'], global_loss, results, i)
            self.save_results('Train', dataset['test'], global_loss, results)

    def test_one_epoch(self, model, dataset):
        model.eval()
        with torch.no_grad():
            results = [0 for _ in range(len(self.metrics))]
            global_loss = 0
            for i, (data, target) in enumerate(dataset['test']):
                if self.debug:
                    if i != 0:
                        break
                data, target = data.to(self.device), target.to(self.device)

                output = model(data)
                loss = self.criterion(output, target.long())
                for k, m in enumerate(self.metrics):
                    results[k] += m(output, target)
                global_loss += loss.item()

                self.progress_message('Test', dataset['test'], global_loss, results, i)
                self.save_results('Test', dataset['test'], global_loss, results)

    def save_results(self, name, dataset, loss, metrics_results):
        with open(self.save_path + '_results.txt', 'a') as f:
            message = f'{name}: epoch {self.current_epoch + 1}/{self.epochs} -> '
            loss = float(loss) / (len(dataset) * dataset.batch_size)
            message += f'{self.criterion.name} loss = {loss}, '
            for k, m in enumerate(self.metrics):
                message += f'{m.name} = {float(metrics_results[k]) / (len(dataset) * dataset.batch_size)} '
            message += '\n'
            f.write(message)


OBJECT = StandardTrainer
