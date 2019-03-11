import numpy as np 
import optimizers

class Trainer(object):
    def __init__(self, model, data=None, **kwargs):
        #The neural network model
        self.model = model
        self.loss_history = []
        if data is not None:
            self.X_train = data['X_train']
            self.y_train = data['y_train']
            self.X_val = data['X_val']
            self.y_val = data['y_val']

           

        #kwargs
        self.optimizer = kwargs.pop('update_rule', 'adam')
        self.optimizer_config = kwargs.pop('optim_config', {})
        self.learning_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)


        if not hasattr(optimizers, self.optimizer):
            raise ValueError('Invalid update rule {}'.format(self.optimizer))
        else:
            self.optimizer = getattr(optimizers, self.optimizer)
            print('optimizer set to {}.'.format(self.optimizer))

        self._reset()


    def get_losshistory(self):
        return self.loss_history

    def get_train_acc_history(self):
        return self.train_acc_history

    def get_val_acc_history(self):
        return self.val_acc_history

    def _reset(self):
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        self.optimizer_config = {}
        for param in self.model.params:
            d = {k: v for k, v in self.optimizer_config.items()}
            self.optimizer_config[param] = d

    
    def accuracy(self, X, y, num_samples=None, batch_size=100):
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = int(N / batch_size)
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []

        #for i in range(num_batches):
            #start = i * batch_size
            #end = (i + 1) * batch_size
            #scores = self.model.loss(X[start:end])
            #y_pred.append(np.argmax(scores, axis=1))


        scores = self.model.loss(X)
        y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)


        acc = np.mean(y_pred == y)

        true_pos  = sum((y_pred == 1) & (y == 1))
        false_pos = sum((y_pred == 1) & (y == 0))
        false_neg = sum((y_pred == 0) & (y == 1))

        precision 	= true_pos * 1.0 / (true_pos + false_pos)
        recall  	= true_pos * 1.0 / (true_pos + false_neg)

        f1 = (2 * precision * recall) / (precision + recall)

        return acc, f1, precision, recall

    def _step(self):

        #number of training examples
        n = self.X_train.shape[0]

        #Making a mini batch
        batch_mask = np.random.choice(n, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)
           

        for p, w in self.model.params.items():
            dW = grads[p]
            config = self.optimizer_config[p]
            next_W, next_config = self.optimizer(w, dW, config)
            self.model.params[p] = next_W
            self.optimizer_config[p] = next_config

    


    def train(self):
        """
        Run optimization to train the model.
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train / self.batch_size, 1)
        num_iterations = int(self.num_epochs * iterations_per_epoch)

        for t in range(num_iterations):
            self._step()

            # Maybe print training loss
            if self.verbose and t % self.print_every == 0:
                print ('(Iteration %d / %d) loss: %f' % (t + 1, num_iterations, self.loss_history[-1]))

            # At the end of every epoch, increment the epoch counter and decay the
            # learning rate.
            epoch_end = (t % 1000) == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optimizer_config:
                    self.optimizer_config[k]['learning_rate'] *= self.learning_decay

            first_it = (t == 0)
            last_it = (t == num_iterations + 1)
            if first_it or last_it or epoch_end or t%500==0:
                train_acc, _, _, _ = self.accuracy(self.X_train, self.y_train,
                                                num_samples=1000)
                val_acc, f1, prec, rec = self.accuracy(self.X_val, self.y_val)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)

                if self.verbose:
                    print ('(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                            self.epoch, self.num_epochs, train_acc, val_acc))

                    print("F1: {} \t Precision {} \t recall {}".format(f1, prec, rec))
                # Keep track of the best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # At the end of training swap the best params into the model
        self.model.params = self.best_params
        print(self.best_val_acc)
        return train_acc, val_acc, f1, prec, rec

