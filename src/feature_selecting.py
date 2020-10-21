import abc
import pandas as pd
import numpy as np
import multiprocessing as mp
import time


class FeatureSelector(abc.ABC):

    @abc.abstractmethod
    def run_selection(self, data: pd.DataFrame, label, prediction_function):
        """

        :param data:
        :param prediction_function: function that takes the data and label,
            calculates the model and return a measure. IMPORTANT!
            This is a maximizer. Therefore the measurement should be edited to express an improvement when the
            value is increasing
        :return:
        """
        pass


class CorrelationSelector(FeatureSelector):
    """
    This FeatureSelector removes one feature in each run until the
    """
    def __init__(self, squared: bool = False):
        self.squared = squared

    def run_selection(self, data: pd.DataFrame, label, prediction_function, early_stopping_iter: int = 5,
                      tolerance: float = 0.001, verbose: bool = True):
        chosen_columns = []
        frozen_measures = []
        max_measure = -np.Inf
        iterations_not_improved = 0

        converged = False
        i = 0
        corr_data = data.copy()
        while not converged:
            print("Run: {}".format(i))
            corrs = corr_data.corr()
            if self.squared:
                corrs = np.sqrt(corrs)
            sum_corrs = corrs.sum(axis=1)
            min_corr = corrs.index[np.nanargmin(np.abs(sum_corrs))]
            tmp = data[chosen_columns + [min_corr]]
            measure = prediction_function(tmp, label)

            if measure > max_measure:
                diff = (measure - max_measure) / max_measure
                if diff > tolerance:
                    iterations_not_improved = 0
                max_measure = measure
            else:
                iterations_not_improved += 1

            corr_data.drop(min_corr, axis=1, inplace=True)
            chosen_columns.append(min_corr)
            frozen_measures.append(measure)

            if verbose:
                print("Chosen column: {} - with Measure value: {}".format(chosen_columns[-1], frozen_measures[-1]))
            if iterations_not_improved >= early_stopping_iter:
                converged = True
            i += 1
            print(frozen_measures)

        col_index = int(np.argmax(frozen_measures))
        return chosen_columns[:col_index], frozen_measures[:col_index]


class GreedyForwardSelector(FeatureSelector):
    """
    calculated the best improvement - most error reduction for each feature and adds this to the feature subset.
    The Selector stops when no improvement is made since X iterations and chooses the iteration with the best result
    """
    def run_selection(self, data: pd.DataFrame, label, prediction_function, early_stopping_iter: int = 5,
                      tolerance: float = 0.001, verbose: bool = True, max_processes: int = 8):
        """

        :param data:
        :param prediction_function:
        :param early_stopping_iter: number of iterations after which the selector stops when the measure doesnt improve
        :param tolerance: relative improvement that the next iteration has to achieve in the last early_stopping_iter.
            Otherwise the selector stops running
        :params verbose: whether to print the recent run or not
        :return:
        """
        remaining_columns = data.columns.tolist()
        chosen_columns = []
        frozen_measures = []
        max_measure = -np.Inf
        iterations_not_improved = 0

        converged = False
        i = 0
        while not converged:
            print("Run: {}".format(i))
            if max_processes > 1:
                measures = self.calculate_parallel(chosen_columns, remaining_columns, prediction_function, data, label,
                                              max_processes)
            else:
                measures = self.calculate_sequential(chosen_columns, remaining_columns, prediction_function, data, label)
            col_index = int(np.argmax(measures))
            chosen_measure = measures[col_index]
            if chosen_measure > max_measure:
                diff = (chosen_measure - max_measure) / max_measure
                if diff > tolerance:
                    iterations_not_improved = 0
                max_measure = chosen_measure
            else:
                iterations_not_improved += 1

            frozen_measures.append(max_measure)
            chosen_columns.append(remaining_columns.pop(col_index))
            if verbose:
                print("Chosen column: {} - with Measure value: {}".format(chosen_columns[-1], frozen_measures[-1]))
            if iterations_not_improved >= early_stopping_iter:
                converged = True
            i += 1
            print(frozen_measures)

        col_index = int(np.argmax(frozen_measures))
        return chosen_columns[:col_index], frozen_measures[:col_index]

    @staticmethod
    def calculate_parallel(chosen_columns, remaining_columns, prediction_function, data, label, max_processes):
        queues = [mp.Queue() for _ in range(len(remaining_columns))]
        processes = []
        semaphore = mp.Semaphore(max_processes)
        for i, col in enumerate(remaining_columns):
            tmp = data[chosen_columns + [col]]
            p = mp.Process(target=task, args=(prediction_function, tmp, label, col, queues[i], semaphore))
            processes.append(p)

        for p in processes:
            while True:
                if semaphore.acquire(timeout=1):
                    break
            p.start()

        measures_cols = dict([q.get() for q in queues])

        for p in processes:
            p.join()

        return [measures_cols[col] for col in remaining_columns]

    @staticmethod
    def calculate_sequential(chosen_columns, remaining_columns, prediction_function, data, label):
        measures = []
        for col in remaining_columns:
            m = prediction_function(data[chosen_columns + [col]], label)
            print(m)
            measures.append(m)
        return measures


def task(prediction_function, data, label, col, queue, semaphore):
    measure = prediction_function(data, label)
    queue.put((col, measure))
    semaphore.release()
    time.sleep(0.01)


if __name__ == "__main__":
    def optim_func(data):
        return 1 / data.shape[1] + np.random.rand(1)

