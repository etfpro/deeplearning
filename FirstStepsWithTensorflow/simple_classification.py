import tensorflow as tf
import numpy as np

# 훈련용 데이터
# - [털 유무, 날개 유무]
# 레이블(One-hot encoding)
# - 기타: [1, 0, 0]
# - 포유류: [0, 1, 0]
# - 조류: [0, 0, 1]

# 읽어들인 데이터는 행과 열이 바뀌어 있다
data = np.loadtxt('./data.csv', delimiter=',', unpack=False, dtype='float32')
x_data = data[:, 0:2] # [털, 날개] 특성 추출
t_data = data[:, 2:] # 레이블 추출(one-hot encoding)

# 입력, 레이블 데이터를 위한 placeholder
X = tf.placeholder(tf.float32)
t = tf.placeholder(tf.float32)


# 가중치 초기화(입력 특성 2, 출력의 클래스 3) 및 가중합-활성화함수 연산
with tf.name_scope('layer1'): # 텐서보드 layer1 계층
    W1 = tf.Variable(tf.random_uniform([2, 10], -1.0, 1.0), name='W1')
    L1 = tf.nn.relu(tf.matmul(X, W1))

with tf.name_scope('layer2'): # 텐서보드 layer2 계층
    W2 = tf.Variable(tf.random_uniform([10, 20], -1.0, 1.0), name='W2')
    L2 = tf.nn.relu(tf.matmul(L1, W2))

with tf.name_scope('output'): # 텐서보드 output 계층
    W3 = tf.Variable(tf.random_uniform([20, 3], -1.0, 1.0), name='W3')
    model = tf.matmul(L2, W3)


with tf.name_scope('optimizer'): # 텐서보드 optimizer 계층
    # 손실함수: 크로스엔트로피 에러
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t, logits=model))

    # 훈련 연산 준비
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    global_step = tf.Variable(0, trainable=False, name='global_step') # 학습 횟수를 카운트하는 변수
    train_op = optimizer.minimize(cost, global_step)

    # 손실값을 추적하기 위해 수집할 값 지정
    tf.summary.scalar("cost", cost)

    tf.summary.histogram("Weights1", W1)
    tf.summary.histogram("Weights2", W2)
    tf.summary.histogram("Weights3", W3)


# 텐서플로 세션 초기화
sess = tf.Session()

# 위에 정의한 변수들을 저장하기 위한 saver 생성
saver = tf.train.Saver(tf.global_variables())

# 기존에 학습된 모델 파일(체크포인트 파일)이 있는지 점검
ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path) # 기존에 학습된 모델을 읽어온다.
else:
    sess.run(tf.global_variables_initializer())


# 앞서 지정한 텐서들을 전부 수집
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs", sess.graph) # 그래프와 텐서들의 값을 저장할 디렉터리 설정


# 학습 진행
for step in range(100):
    # train과 cost 두개의 연산을 동시에 수행
    _, cost_val = sess.run([train_op, cost], feed_dict={X:x_data, t:t_data})
    print("Step {}: cost - {}".format(sess.run(global_step), cost_val))

    # 각 학습 단계별로 값들을 수집하고 저장
    summary = sess.run(merged,  feed_dict={X:x_data, t:t_data})
    writer.add_summary(summary, global_step=sess.run(global_step))

# 학습된 변수들을 지정한 체크포인트 파일에 저장
saver.save(sess, './model/dnn.ckpt', global_step=global_step)


# 예측
prediction = tf.argmax(model, axis=1)
target = tf.argmax(t, axis=1)

print("예측값:", sess.run(prediction, feed_dict={X:x_data}))
print("정답값:", sess.run(target, feed_dict={t:t_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) # true, false를 1, 0으로 변환 후, 평균을 낸다.
print("정확도: %.2f" % sess.run(accuracy * 100, feed_dict={X: x_data, t:t_data}))

print("Global Step:", sess.run(global_step))
sess.close()