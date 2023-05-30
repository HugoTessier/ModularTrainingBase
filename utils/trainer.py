import torch
import sys
import os


class Trainer:
    '''
    Contains all the necessary to train a given network, on a given dataset, save the progression of the training and
    resume to its previous state if it gets interrupted and relaunched.
    '''

    def __init__(self, epochs, optimizer, scheduler, criterion, metrics, device, output_path, name, debug=False):
        '''
        Creates the trainer.

        :param epochs: Maximum number of epochs of training.
        :param optimizer: Optimizer (SGD, Adam, etc.) to update the network.
        :param scheduler: Scheduler (MultiStepLR, Cosine, etc.) to set the learning rate.
        :param criterion: Criterion (CrossEntropy, RMI, etc.) that defines the loss function.
        :param metrics: List of metrics (Top-1, Top-5, MIOU, etc.). These metrics must be classes, that contain a "name"
        attribute to give its name, and a method "__call__" that takes the prediction and the target as input, and
        returns the said metric as a float number.
        :param device: Device on which to run the training (cpu, cuda, etc.).
        :param output_path: Folder in which to save the results and save file of the training.
        :param name: Base name that will define the name of the save files (name_state.pt, name_model.pt, etc.).
        :param debug: When set to True, interrupts each epoch after the first batch, so that the whole training loop
        can be tested easily.
        '''
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

    def _save_state(self, model):
        with open(self.save_path + '_state.pt', 'wb') as f:
            torch.save({'current_epoch': self.current_epoch}, f)
        with open(self.save_path + '_model.pt', 'wb') as f:
            torch.save(model.state_dict(), f)
        with open(self.save_path + '_optimizer.pt', 'wb') as f:
            torch.save(self.optimizer.state_dict(), f)

    def _load_state(self, model):
        with open(self.save_path + '_state.pt', 'rb') as f:
            self.current_epoch = torch.load(f)['current_epoch']
            self.scheduler.last_epoch = self.current_epoch
        with open(self.save_path + '_model.pt', 'rb') as f:
            model.load_state_dict(torch.load(f))
        with open(self.save_path + '_optimizer.pt', 'rb') as f:
            self.optimizer.load_state_dict(torch.load(f))

    def _need_to_resume_previous_state(self):
        if os.path.exists(self.save_path + '_state.pt'):
            with open(self.save_path + '_state.pt', 'rb') as f:
                return self.current_epoch < torch.load(f)['current_epoch']
        else:
            return False

    def __call__(self, model, dataset):
        '''
        Launches a complete training of the model on the dataset. Resets the learning rate at the beginning, so that
        launching this method several times equals a warm restart.

        :param model: A Pytorch model to train.
        :param dataset: A dict that contains two DataLoaders called "train" and "test".
        '''
        self._training_start_routine(model, dataset)
        while self.current_epoch < self.epochs:
            self._epoch_start_routine(model, dataset)
            self._train_one_epoch(model, dataset)
            self._epoch_mid_routine(model, dataset)
            self._test_one_epoch(model, dataset)
            self._epoch_end_routine(model, dataset)

    def train_n_epochs(self, model, dataset, n):
        """
        Trains the model, on the dataset, during n epochs. This allows to split a same training into several steps.

        :param model: A Pytorch model to train.
        :param dataset: A dict that contains two DataLoaders called "train" and "test".
        :param n: The number of epochs to train.
        """
        if self.current_epoch == 0:
            self._training_start_routine(model, dataset)
        for _ in range(n):
            if self.current_epoch < self.epochs:
                self._epoch_start_routine(model, dataset)
                self._train_one_epoch(model, dataset)
                self._epoch_mid_routine(model, dataset)
                self._test_one_epoch(model, dataset)
                self._epoch_end_routine(model, dataset)

    def _epoch_start_routine(self, model, dataset):
        if self._need_to_resume_previous_state():
            self._load_state(model)
        else:
            self._save_state(model)
        print(f'Epoch {self.current_epoch + 1}/{self.epochs}')

    def _epoch_mid_routine(self, model, dataset):
        print()

    def reset_training(self):
        '''
        Resets the counter of elapsed epochs and the learning rate, so that a completely new training can be started.
        '''
        self.current_epoch = 0
        self.scheduler.last_epoch = -1
        self.scheduler.step()

    def _training_start_routine(self, model, dataset):
        print(self.name)
        self.current_epoch = 0
        self.scheduler.last_epoch = -1
        self.scheduler.step()
        self.save_path = os.path.join(self.output_path, self.name)
        if not os.path.isdir(self.output_path):
            try:
                os.mkdir(self.output_path)
            except OSError:
                print(f'Failed to create the folder {self.output_path}')
            else:
                print(f'Created folder {self.output_path}')

    def _epoch_end_routine(self, model, dataset):
        print()
        self.scheduler.step()
        self.current_epoch += 1

    def _progress_message(self, name, dataset, loss, metrics_results, current_step):
        message = f'\r{name} ({current_step + 1}/{len(dataset)}) -> '
        rounded_loss = round(float(loss) / (current_step + 1), 3)
        message += f'{self.criterion.name} loss = {rounded_loss}, '
        for k, m in enumerate(self.metrics):
            rounded_metric = round(float(metrics_results[k]) / ((current_step + 1) * dataset.batch_size), 3)
            message += f'{m.name} = {rounded_metric}\t'
        message += '           '
        sys.stdout.write(message)

    def _train_one_epoch(self, model, dataset):
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

            self._progress_message('Train', dataset['train'], global_loss, results, i)
            self._save_results('Train', dataset['test'], global_loss, results)

    def _test_one_epoch(self, model, dataset):
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

                self._progress_message('Test', dataset['test'], global_loss, results, i)
                self._save_results('Test', dataset['test'], global_loss, results)

    def _save_results(self, name, dataset, loss, metrics_results):
        with open(self.save_path + '_results.txt', 'a') as f:
            message = f'{name}: epoch {self.current_epoch + 1}/{self.epochs} -> '
            loss = float(loss) / (len(dataset) * dataset.batch_size)
            message += f'{self.criterion.name} loss = {loss}, '
            for k, m in enumerate(self.metrics):
                message += f'{m.name} = {float(metrics_results[k]) / (len(dataset) * dataset.batch_size)} '
            message += '\n'
            f.write(message)


OBJECT = Trainer
