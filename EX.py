# def get_action(self, obs):
# 		mean = self.actor(obs)
# 		dist = MultivariateNormal(mean, self.cov_mat)
# 		action = dist.sample()
# 		log_prob = dist.log_prob(action)
# 		return action.detach().numpy(), log_prob.detach()

# 	def evaluate(self, batch_obs, batch_acts):
# 		mean = self.actor(batch_obs)
# 		dist = MultivariateNormal(mean, self.cov_mat)
# 		log_probs = dist.log_prob(batch_acts)
# 		return log_probs