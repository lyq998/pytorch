from evolve import Evolve_CNN
from utils import *


def begin_evolve(m_prob, m_eta, x_prob, x_eta, pop_size, batch_size, total_generation_number):
    cnn = Evolve_CNN(m_prob, m_eta, x_prob, x_eta, pop_size, batch_size)
    cnn.initialize_popualtion()
    cnn.evaluate_fitness(0)
    for cur_gen_no in range(total_generation_number):
        print('The {}/{} generation'.format(cur_gen_no + 1, total_generation_number))
        cnn.recombinate(cur_gen_no + 1)
        cnn.environmental_selection(cur_gen_no + 1)


def restart_evolve(m_prob, m_eta, x_prob, x_eta, pop_size, batch_size, total_gene_number):
    gen_no, pops, _ = load_population()
    cnn = Evolve_CNN(m_prob, m_eta, x_prob, x_eta, pop_size, batch_size)
    cnn.pops = pops
    if gen_no < 0:  # go to evaluate
        print('first to evaluate...')
        cnn.evaluate_fitness(1)
    else:
        # 判断有没有经历environmental_selection
        if pops.get_pop_size() == pop_size * 2:
            cur_gen_no = gen_no
            cnn.environmental_selection(cur_gen_no)
        for cur_gen_no in range(gen_no + 1, total_gene_number + 1):
            print('Continue to evolve from the {}/{} generation...'.format(cur_gen_no, total_gene_number))
            cnn.recombinate(cur_gen_no)
            cnn.environmental_selection(cur_gen_no)


if __name__ == '__main__':
    # train_data, validation_data, test_data = get_mnist_data()
    batch_size = 10
    total_generation_number = 20  # total generation number
    pop_size = 10

    # 测试
    gen_no, pops, create_time = load_population()
    cnn = Evolve_CNN(0.9, 1, 0.2, 1, pop_size, batch_size)
    cnn.pops = pops
    cur_gen_no = 1
    cnn.recombinate(cur_gen_no)

    # print(gen_no, pops, create_time)
    # print(pops.get_pop_size())

    # begin_evolve(0.9, 1, 0.2, 1, pop_size, batch_size, total_generation_number)
    # restart_evolve(0.9, 1, 0.2, 1, pop_size, batch_size, total_generation_number)
