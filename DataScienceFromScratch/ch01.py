# ch01.py
from __future__ import division
from collections import Counter

users = [
    {"id": 0, "name": "Hero"},
    {"id": 1, "name": "Dunn"},
    {"id": 2, "name": "Sue"},
    {"id": 3, "name": "Chi"},
    {"id": 4, "name": "Thor"},
    {"id": 5, "name": "Clive"},
    {"id": 6, "name": "Hicks"},
    {"id": 7, "name": "Devin"},
    {"id": 8, "name": "Kate"},
    {"id": 9, "name": "Klein"}
    ]


friendships = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
               (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

for user in users:
    user["friends"] = []

# 각 사용자의 친구 연결
for i, j in friendships:
    users[i]["friends"].append(users[j])
    users[j]["friends"].append(users[i])


# 지정한 사용자의 친구의 수 계산
def number_of_friends(user):
    return len(user["friends"])

# 각 사용자의 id와 친구의 수를 나타내는 튜플의 리스트
num_friends_by_id = [(user["id"], number_of_friends(user))
                     for user in users]

# 친구의 수가 많은 순서대로 각 사용자의 id를 정렬
sorted_friends = sorted(num_friends_by_id,
       key=lambda num_friend: num_friend[1],
       reverse=True)


################################################################################
# page 6
################################################################################

# 지정한 사용자의 친구의 친구의 id 리스트를 구한다(중복 포함)
def friends_of_friend_ids_bad(user):
    return [foaf["id"]
            for friend in user["friends"]
            for foaf in friend["friends"]]
"""
for user in users:
    print("%s[%d]'s friends: %s" %(user["name"],
                                   user["id"],
                                   [(friend["name"], friend["id"]) for friend in user["friends"]])
          )
"""

################################################################################
# page 7
################################################################################

# 지정한 두 사용자가 서로 다른 사용자 인지
def not_the_same(user1, user2):
    return user1["id"] != user2["id"]


# 지정한 사용자(user)의 친구 중에 other_user가 없는지 검사
def not_friends(user, other_user):
    return all(not_the_same(other_user, friend)
               for friend in user["friends"])


# 지정한 사용자의 친구의 친구 중 이미 아는 사람을 제외한 사람들의 id 수를 구한다
def friends_of_friend_ids(user):
    return Counter(foaf["id"]
                   for friend in user["friends"]
                   for foaf in friend["friends"]
                   if not_the_same(user, foaf)
                   and not_friends(user, foaf))



interests = [
    (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"), (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
    (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"), (1, "Postgres"),
    (2, "Python"), (2, "scikit-learn"), (2, "scipy"), (2, "numpy"), (2, "statsmodels"), (2, "pandas"),
    (3, "R"), (3, "Python"), (3, "statistics"), (3, "regression"), (3, "probability"),
    (4, "machine learning"), (4, "regression"), (4, "decision trees"), (4, "libsvm"),
    (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"), (5, "Haskell"), (5, "programming languages"),
    (6, "statistics"), (6, "probability"), (6, "mathematics"), (6, "theory"),
    (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"), (7, "neural networks"),
    (8, "neural networks"), (8, "deep learning"), (8, "Big Data"), (8, "artificial intelligence"),
    (9, "Hadoop"), (9, "Java"), (9, "MapReduce"), (9, "Big Data")
]
