from os import path
import matplotlib.pyplot as plt


def read_results(model_path, corpus_name):

    r = []

    with open(path.join(model_path, corpus_name + '.0.eval.out'), 'r') as eval_file:
        lines = eval_file.readlines()
        sure_precision = float(lines[9].strip().split()[2])
        sure_recall = float(lines[10].strip().split()[2])
        sure_fmeasure = float(lines[11].strip().split()[2])
        prob_precision = float(lines[14].strip().split()[2])
        prob_recall = float(lines[15].strip().split()[2])
        prob_fmeasure = float(lines[16].strip().split()[2])
        aer = float(lines[18].strip().split()[2])

        r.append([0,0,0,sure_precision,sure_recall,sure_fmeasure,prob_precision,prob_recall,prob_fmeasure,aer])

    with open(path.join(model_path, corpus_name + '.results'), 'r') as results_file:
        lines = results_file.readlines()
        for line in lines:
            split_line = line.strip().split(',')
            run_no = int(split_line[0])
            run_time = float(split_line[1])
            likelihood = float(split_line[2])
            with open(path.join(model_path, corpus_name + '.' + str(run_no) + '.eval.out'), 'r') as eval_file:
                lines = eval_file.readlines()
                sure_precision = float(lines[9].strip().split()[2])
                sure_recall = float(lines[10].strip().split()[2])
                sure_fmeasure = float(lines[11].strip().split()[2])
                prob_precision = float(lines[14].strip().split()[2])
                prob_recall = float(lines[15].strip().split()[2])
                prob_fmeasure = float(lines[16].strip().split()[2])
                aer = float(lines[18].strip().split()[2])

                r.append(
                    [run_no, run_time, likelihood, sure_precision, sure_recall, sure_fmeasure, prob_precision, prob_recall, prob_fmeasure,
                     aer])
    return r


def main():
    """Program entry point"""

    data_path = path.join(path.dirname(__file__), '..', 'data')
    corpus_name = 'hansards.36.2'  # hansards.36.2

    model_paths = [
        path.join(data_path, 'model', 'ibm2', 'ibm1-5'),
        path.join(data_path, 'model', 'ibm2', 'uniform'),
        path.join(data_path, 'model', 'ibm2', 'random1'),
        path.join(data_path, 'model', 'ibm2', 'random2'),
        path.join(data_path, 'model', 'ibm2', 'random3'),
        path.join(data_path, 'model', 'ibm1', 'uniform'),
        path.join(data_path, 'model', 'ibm1', 'random1'),
        path.join(data_path, 'model', 'ibm1', 'random2'),
        path.join(data_path, 'model', 'ibm1', 'random3'),
        path.join(data_path, 'model', 'ibm1', 'random-n0.01'),
        path.join(data_path, 'model', 'ibm1', 'random-n0.005'),
        path.join(data_path, 'model', 'ibm1', 'random-n0.0005'),
        path.join(data_path, 'model', 'ibm1', 'random-q02'),
        path.join(data_path, 'model', 'ibm1', 'random-q03')
    ]

    all_models = []

    for model_path in model_paths:
        name = path.split(path.split(model_path)[0])[1] + ' ' + path.split(model_path)[1]
        all_models.append((name,read_results(model_path, corpus_name)))

    plot_likelihoods(all_models)
    plot_aer(all_models)
    plot_precision(all_models)
    plot_recall(all_models)

    print_best(all_models)


def print_best(all_models):
    print "Best Recall"
    print max([(x[0],x[1][20][7]) for x in all_models], key=lambda x: x[1])
    print "Worst Recall"
    print min([(x[0],x[1][20][7]) for x in all_models], key=lambda x: x[1])
    print "Best Precision"
    print max([(x[0], x[1][20][6]) for x in all_models], key=lambda x: x[1])
    print "Worst Precision"
    print min([(x[0], x[1][20][6]) for x in all_models], key=lambda x: x[1])
    print "Best AER"
    print min([(x[0], x[1][20][9]) for x in all_models], key=lambda x: x[1])
    print "Worst AER"
    print max([(x[0], x[1][20][9]) for x in all_models], key=lambda x: x[1])


def plot_likelihoods(all_models):
    legends = []
    for (name, model_data) in all_models:
        legends.append(name)
        plt.plot(range(1, 21), [iteration_data[2] for iteration_data in model_data if iteration_data[2] != 0])
    plt.legend(legends, loc='lower right', prop={'size':10})
    plt.xlabel('Iterations')
    plt.ylabel('Log-likelihood')
    plt.title('Log-likelihood progression during iterations')
    plt.grid(True)
    plt.savefig("../data/log-likelihood.png")
    plt.clf()


def plot_aer(all_models):
    legends = []
    for (name, model_data) in all_models:
        legends.append(name)
        plt.plot(range(0, 21), [iteration_data[9] for iteration_data in model_data])
    plt.legend(legends, loc='upper right', prop={'size':10})
    plt.xlabel('Iterations')
    plt.ylabel('Alignment Error Rate (AER)')
    plt.title('AER progression during iterations')
    plt.grid(True)
    plt.savefig("../data/aer.png")
    plt.clf()


def plot_precision(all_models):
    legends = []
    for (name, model_data) in all_models:
        legends.append(name)
        plt.plot(range(0, 21), [iteration_data[6] for iteration_data in model_data])
    plt.legend(legends, loc='lower right', prop={'size':10})
    plt.xlabel('Iterations')
    plt.ylabel('Precision')
    plt.title('Precision progression during iterations')
    plt.grid(True)
    plt.savefig("../data/precision.png")
    plt.clf()


def plot_recall(all_models):
    legends = []
    for (name, model_data) in all_models:
        legends.append(name)
        plt.plot(range(0, 21), [iteration_data[7] for iteration_data in model_data])
    plt.legend(legends, loc='lower right', prop={'size':10})
    plt.xlabel('Iteration')
    plt.ylabel('Recall')
    plt.title('Recall progression during iterations')
    plt.grid(True)
    plt.savefig("../data/recall.png")
    plt.clf()


if __name__ == "__main__":
    main()
