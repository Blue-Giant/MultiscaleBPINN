from utilizers import DNN_tools


# 记录字典中的一些设置
def dictionary_out2file(R_dic, log_fileout):
    # -----------------------------------------------------------------------------------------------------------------
    DNN_tools.log_string('PDE type for problem: %s\n' % (R_dic['PDE_type']), log_fileout)
    DNN_tools.log_string('Equation name for problem: %s\n' % (R_dic['equa_name']), log_fileout)
    DNN_tools.log_string('The  dimension of independent variable for problem: %s\n' % (R_dic['indim']), log_fileout)

    # -----------------------------------------------------------------------------------------------------------------
    DNN_tools.log_string('Network model of solving problem: %s\n' % str(R_dic['model']), log_fileout)

    DNN_tools.log_string('Activate function for NN-input: %s\n' % str(R_dic['act_name2Input']), log_fileout)

    DNN_tools.log_string('Activate function for NN-hidden: %s\n' % str(R_dic['act_name2Hidden']), log_fileout)
    DNN_tools.log_string('Activate function for NN-output: %s\n' % str(R_dic['act_name2Output']), log_fileout)

    DNN_tools.log_string('hidden layer:%s\n' % str(R_dic['Two_hidden_layer']), log_fileout)
    DNN_tools.log_string('hidden layer:%s\n' % str(R_dic['Three_hidden_layer']), log_fileout)

    # --------------------------------------------------------------------------------------------------------------
    DNN_tools.log_string('Mode to update the parameters of neural network: %s\n' % str(R_dic['mode2update_para']),
                         log_fileout)
    DNN_tools.log_string('The learning rate to update the parameters of neural network: %s\n' %
                         str(R_dic['learning_rate']), log_fileout)
    if R_dic['update_lr'] is True:
        DNN_tools.log_string('Change the learning rate by step model!!!!!!!\n', log_fileout)
        DNN_tools.log_string('The step of updating learning rate:: %s\n' % str(R_dic['step2update_lr']), log_fileout)
        DNN_tools.log_string('The gamma of updating learning rate:: %s\n' % str(R_dic['gamma2update_lr']), log_fileout)
    else:
        DNN_tools.log_string('The learning rate keeps unchanged!!!!!!!\n', log_fileout)

    DNN_tools.log_string('Mode to generate the training data: %s\n' % str(R_dic['opt2sampling']), log_fileout)
    DNN_tools.log_string('Noise level to interfere the train data: %s\n' % str(R_dic['noise_level']),
                         log_fileout)
    if R_dic['mode2update_para'] == 'PINN':
        if R_dic['activate_stop2pinn'] != 0:
            DNN_tools.log_string('activate the stop_step for PINN and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
        else:
            DNN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']),
                                 log_fileout)
    else:
        DNN_tools.log_string('The total num of hamilton sample = %s\n' % str(R_dic['sample_num2hamilton']), log_fileout)


def print_log_validation(log_probability, log_out=None):
    print(" Expected validation log probability: {:.3f}".format(log_probability))
    DNN_tools.log_string('Expected validation log probability: %s\n\n' % str(log_probability.item()), log_out)


def print_log_errors2force_side(mse2test=0.01, rel2test=0.01, log_out=None):
    # 将运行结果打印出来
    print('mean square error of between force-predict and force-real for testing: %.10f\n' % mse2test)
    print('relative error of of between force-predict and force-real for testing: %.10f\n' % rel2test)

    DNN_tools.log_string('mean square error of between force-predict and force-real for testing: %.10f' % mse2test,
                         log_out)
    DNN_tools.log_string('relative error of between force-predict and force-real for testing: %.10f\n\n' % rel2test,
                         log_out)


def print_log_errors2solution(mse2test=0.01, rel2test=0.01, log_out=None):
    # 将运行结果打印出来
    print('mean square error of between solu-predict and solu-real for testing: %.10f\n' % mse2test)
    print('relative error of of between solu-predict and solu-real for testing: %.10f\n' % rel2test)

    DNN_tools.log_string('mean square error of between solu-predict and solu-real for testing: %.10f' % mse2test, log_out)
    DNN_tools.log_string('relative error of between solu-predict and solu-real for testing: %.10f\n\n' % rel2test, log_out)


def print_log2pre_train_model(log_out=None):
    # 将运行结果打印出来
    print('The results for pre_train model!!!!!!')
    DNN_tools.log_string('The results for pre_train model!!!!!!!!', log_out)


def print_log_mse_rel(mse2solu=0.01, rel2solu=0.01, mse2para=0.01, rel2para=0.01, mse2force=0.01, rel2force=0.01, log_out=None):
    # 将运行结果打印出来
    print('mean square error of mean-predict and exact for solution: %.10f\n' % mse2solu)
    print('relative error of mean-predict and exact for solution: %.10f\n' % rel2solu)

    print('mean square error of mean-predict and exact for parameter: %.10f\n' % mse2para)
    print('relative error of mean-predict and exact for parameter: %.10f\n' % rel2para)

    print('mean square error of mean-predict and exact for force: %.10f\n' % mse2force)
    print('relative error of mean-predict and exact for force: %.10f\n' % rel2force)

    DNN_tools.log_string('mean square error of mean-predict and exact for solution: %.10f' % mse2solu, log_out)
    DNN_tools.log_string('relative error of mean-predict and exact for solution: %.10f\n\n' % rel2solu, log_out)

    DNN_tools.log_string('mean square error of mean-predict and exact for parameter: %.10f' % mse2para, log_out)
    DNN_tools.log_string('relative error of mean-predict and exact for parameter: %.10f\n\n' % rel2para, log_out)

    DNN_tools.log_string('mean square error of mean-predict and exact for force: %.10f' % mse2force, log_out)
    DNN_tools.log_string('relative error of mean-predict and exact for force: %.10f\n\n' % rel2force, log_out)