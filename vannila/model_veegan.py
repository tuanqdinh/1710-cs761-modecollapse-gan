
get_ipython().magic(u'pylab inline')
import tensorflow as tf
import itertools
import seaborn as sns
slim = tf.contrib.slim
ds = tf.contrib.distributions
st = tf.contrib.bayesflow.stochastic_tensor
graph_replace = tf.contrib.graph_editor.graph_replace


params = {
	'batch_size': 500,
	'latent_dim': 2,
	'eps_dim': 1,
	'input_dim': 254,
	'n_layer_disc': 2,
	'n_hidden_disc': 128,
	'n_layer_gen': 2,
	'n_hidden_gen': 128,
	'n_layer_inf': 2,
	'n_hidden_inf': 128,
}


def create_distribution(batch_size, num_components=25, num_features=2, **kwargs):
	cat = ds.Categorical(tf.zeros(num_components, dtype=float32))
	mus = np.array([np.array([i, j]) for i, j in itertools.product(range(-4, 5, 2),
	range(-4, 5, 2))], dtype=float32)
	s = 0.05
	sigmas = [np.array([s, s]).astype(float32) for i in range(num_components)]
	components = list((ds.MultivariateNormalDiag(mu, sigma)
					   for (mu, sigma) in zip(mus, sigmas)))
	data = ds.Mixture(cat, components)
	return data.sample(batch_size)

# Network definitions


def standard_normal(shape, **kwargs):
	"""Create a standard Normal StochasticTensor."""
	return st.StochasticTensor(
		ds.MultivariateNormalDiag(mu=tf.zeros(shape), diag_stdev=tf.ones(shape), **kwargs))


def normal_mixture(shape, **kwargs):
	return create_distribution(shape[0], 25, shape[1], **kwargs)


def generative_network(batch_size, latent_dim, input_dim, n_layer, n_hidden, eps=1e-6, X=None):
	with tf.variable_scope("generative"):
		z = normal_mixture([batch_size, latent_dim], name="p_z")
		h = slim.fully_connected(z, n_hidden, activation_fn=tf.nn.relu)
		h = slim.fully_connected(h, n_hidden, activation_fn=tf.nn.relu)
		p = slim.fully_connected(h, input_dim, activation_fn=None)
		x = st.StochasticTensor(
			ds.Normal(p * tf.ones(input_dim), 1 * tf.ones(input_dim), name="p_x"))
	return [x, z]


def inference_network(x, latent_dim, n_layer, n_hidden, eps_dim):
	eps = standard_normal(
		[x.get_shape().as_list()[0], eps_dim], name="eps").value()
	h = tf.concat([x, eps], 1)
	with tf.variable_scope("inference"):
		h = slim.fully_connected(h, n_hidden, activation_fn=tf.nn.relu)
		h = slim.fully_connected(h, n_hidden, activation_fn=tf.nn.relu)
		z = slim.fully_connected(
			h, latent_dim, activation_fn=None, scope="q_z")
	return z


def data_network(x, z, n_layers=2, n_hidden=128, activation_fn=None):
	h = tf.concat([x, z], 1)
	with tf.variable_scope('discriminator'):
		h = slim.fully_connected(h, n_hidden, activation_fn=tf.nn.relu)
		log_d = slim.fully_connected(h, 1, activation_fn=activation_fn)
	return tf.squeeze(log_d, squeeze_dims=[1])

# Construct model and training ops


tf.reset_default_graph()

x = tf.random_normal([params['batch_size'], params['input_dim']])

p_x, p_z = generative_network(params['batch_size'], params['latent_dim'], params['input_dim'],
							  params['n_layer_gen'], params['n_hidden_gen'])

q_z = inference_network(x, params['latent_dim'], params['n_layer_inf'], params['n_hidden_inf'],
						params['eps_dim'])


log_d_prior = data_network(p_x, p_z, n_layers=params['n_layer_disc'],
						   n_hidden=params['n_hidden_disc'])
log_d_posterior = data_network(x, q_z, n_layers=params['n_layer_disc'],
							   n_hidden=params['n_hidden_disc'])


disc_loss = tf.reduce_mean(
	tf.nn.sigmoid_cross_entropy_with_logits(logits=log_d_posterior, labels=tf.ones_like(log_d_posterior)) +
	tf.nn.sigmoid_cross_entropy_with_logits(logits=log_d_prior, labels=tf.zeros_like(log_d_prior)))


recon_likelihood_prior = p_x.distribution.log_prob(x)
recon_likelihood = tf.reduce_sum(graph_replace(
	recon_likelihood_prior, {p_z: q_z}), [1])


gen_loss = tf.reduce_mean(log_d_posterior) - tf.reduce_mean(recon_likelihood)

qvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "inference")
pvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generative")
dvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
opt = tf.train.AdamOptimizer(1e-3, beta1=.5)

train_gen_op = opt.minimize(gen_loss, var_list=qvars + pvars)
train_disc_op = opt.minimize(disc_loss, var_list=dvars)


# In[6]:

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


# Train model

# In[7]:

from tqdm import tqdm
import collections
fs = []

total_batch = 1000

#  Training cycle
for epoch in tqdm(xrange(100)):
	xx = np.vstack([sess.run(q_z) for _ in range(5)])
	yy = np.vstack([sess.run(p_z) for _ in range(5)])
	fig_ = figure(figsize=(5, 5), facecolor='w')

	scatter(xx[:, 0], xx[:, 1],
			edgecolor='none', alpha=0.5)
	scatter(yy[:, 0], yy[:, 1], c='g', edgecolor='none')
	show()


#     Loop over all batches
	for i in range(total_batch):
		_ = sess.run([[gen_loss, disc_loss], train_gen_op, train_disc_op])


# In[8]:

'''Sample 2500 points'''
xx = np.vstack([sess.run(q_z) for _ in range(5)])
yy = np.vstack([sess.run(p_z) for _ in range(5)])


'''KDE Plots'''
sns.set(font_scale=2)
f, (ax1, ax2) = plt.subplots(2, figsize=(10, 15))
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
sns.kdeplot(xx[:, 0], xx[:, 1], cmap=cmap, ax=ax1,
			n_levels=100, shade=True, clip=[[-6, 6]] * 2)
sns.kdeplot(yy[:, 0], yy[:, 1], cmap=cmap, ax=ax2,
			n_levels=100, shade=True, clip=[[-6, 6]] * 2)


'''Evaluation'''
MEANS = np.array([np.array([i, j]) for i, j in itertools.product(range(-4, 5, 2),
																 range(-4, 5, 2))], dtype=float32)
l2_store = []
for x_ in xx:
	l2_store.append([np.sum((x_ - i)**2) for i in MEANS])

mode = numpy.argmin(l2_store, 1).flatten().tolist()
dis_ = [l2_store[j][i] for j, i in enumerate(mode)]
mode_counter = [mode[i]
				for i in range(len(mode)) if numpy.sqrt(dis_[i]) <= 0.15]

print 'Number of Modes Captured: ', len(collections.Counter(mode_counter))
print 'Number of Points Falling Within 3 std. of the Nearest Mode ', numpy.sum(collections.Counter(mode_counter).values())
