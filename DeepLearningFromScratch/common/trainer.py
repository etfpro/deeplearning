from common.optimizer import *


# 신경망 훈련을 대신 해주는 클래스
class Trainer:
    # network: 신경망 클래스
    # x_train: 훈련 데이터
    # t_train: 훈련 데이터의 정답
    # x_test: 테스트 데이터
    # t_test: 테스트 데이터 정답
    # epoch: 주기
    # mini_batch_size: 미니배치 크기
    # evaluate_sample_num_per_epoch
    # verbose: 훈련 결과를 화면에 표시할지 여부
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, batch_size=100,
                 evaluate_sample_num_per_epoch=None, verbose=True):
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.network = network
        self.epochs = epochs
        self.batch_size = batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch
        self.verbose = verbose

        # 훈련 데이트 크기
        self.train_size = x_train.shape[0]

        # 주기 당 훈련 반복 회수
        self.iter_per_epoch = max(self.train_size / batch_size, 1)

        # 최대 훈련 반복 회수
        self.max_iter = int(epochs * self.iter_per_epoch)

        self.train_loss_list = [] # 매 훈련마다 손실값 기록
        self.train_acc_list = [] # 매 주기마다 훈련데이터 정확도 기록
        self.test_acc_list = [] # 매 주기마다 테스트데이터 정확도 기록



    # 훈련
    def train(self):

        cur_epoch = 0
        for i in range(self.max_iter):
            # 미니배치 획득
            batch_mask = np.random.choice(self.train_size, self.batch_size)
            x_batch = self.x_train[batch_mask]
            t_batch = self.t_train[batch_mask]

            # 네트워크로 훈련
            self.network.train(x_batch, t_batch)

            # 손실값 계산
            loss = self.network.lossValue
            self.train_loss_list.append(loss)
            if self.verbose:
                print("train loss:" + str(loss))


            # 매 주기마다 훈련데이터와 테스트데이터의 정확도 측정
            if i % self.iter_per_epoch == 0:
                x_train_sample, t_train_sample = self.x_train, self.t_train
                x_test_sample, t_test_sample = self.x_test, self.t_test
                if self.evaluate_sample_num_per_epoch is not None:
                    t = self.evaluate_sample_num_per_epoch
                    x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                    x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]

                # 훈련데이터 정확도 측덩
                train_acc = self.network.accuracy(x_train_sample, t_train_sample)
                self.train_acc_list.append(train_acc)

                # 테스트데이터 정확도 측정
                test_acc = self.network.accuracy(x_test_sample, t_test_sample)
                self.test_acc_list.append(test_acc)

                if self.verbose:
                    print("=== epoch %000d: Train accuracy = %2.2f%%, Test accuracy = %2.2f%% ===" %
                          (cur_epoch, train_acc * 100.0, test_acc * 100.0))

                cur_epoch += 1


        # 최종 훈련 후, 테스트 데이터에 대한 정확도 측정햐여 표시
        if self.verbose:
            test_acc = self.network.accuracy(self.x_test, self.t_test)
            print("=============== Final Test Accuracy ===============")
            print("%2.2f%%", test_acc * 100.0)




if __name__ == '__main__':
    pass

