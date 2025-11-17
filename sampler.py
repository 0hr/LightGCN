import random

import numpy as np


class Sampler:
    def __init__(self, num_users, num_items, user_pos, neg_strategy="uniform"):
        self.num_users = num_users
        self.num_items = num_items
        self.user_pos = user_pos
        self.neg_strategy = neg_strategy
        item_pop = np.zeros(num_items, dtype=np.float64)
        for items in user_pos.values():
            for i in items:
                item_pop[i] += 1
        item_pop = item_pop + 1e-8
        self.item_pop = item_pop / item_pop.sum()

    def sample_batch(self, batch_size):
        users = np.random.choice(list(self.user_pos.keys()), size=batch_size)
        pos = []
        neg = []
        for u in users:
            if not self.user_pos[u]:
                pos.append(0)
                neg.append(0)
                continue
            pos_i = random.choice(list(self.user_pos[u]))
            pos.append(pos_i)
            neg.append(self._sample_negative(u))
        return np.array(users), np.array(pos), np.array(neg)

    def _sample_negative(self, user):
        if len(self.user_pos[user]) >= self.num_items:
            return np.random.randint(0, self.num_items)
        for _ in range(50):
            if self.neg_strategy == "popularity":
                neg = np.random.choice(self.num_items, p=self.item_pop)
            else:
                neg = np.random.randint(0, self.num_items)
            if neg not in self.user_pos[user]:
                return neg
        return np.random.randint(0, self.num_items)