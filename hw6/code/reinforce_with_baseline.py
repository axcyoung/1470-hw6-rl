import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class ReinforceWithBaseline(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The ReinforceWithBaseline class that inherits from tf.keras.Model.

        The forward pass calculates the policy for the agent given a batch of states. During training,
        ReinforceWithBaseLine estimates the value of each state to be used as a baseline to compare the policy's
        performance with.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(ReinforceWithBaseline, self).__init__()
        self.num_actions = num_actions

        # TODO: Define actor network parameters, critic network parameters, and optimizer
        self.learning_rate = .001
        self.actor1 = tf.keras.layers.Dense(state_size*32)
        self.actor2 = tf.keras.layers.Dense(state_size*16, activation='relu')
        self.actor3 = tf.keras.layers.Dense(self.num_actions)
        
        self.critic1 = tf.keras.layers.Dense(state_size*16)
        self.critic2 = tf.keras.layers.Dense(1)

    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        for each state in the episode
        """
        # TODO: implement this!
        states = tf.convert_to_tensor(states)
        a1 = self.actor1(states)
        a2 = self.actor2(a1)
        a3 = self.actor3(a2)
        probs = tf.nn.softmax(a3)
        return probs

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An [episode_length, state_size] dimensioned array representing the history of states
        of an episode.
        :return: A [episode_length] matrix representing the value of each state.
        """
        # TODO: implement this :D
        states = tf.convert_to_tensor(states)
        c1 = self.critic1(states)
        c2 = self.critic2(c1)
        return c2

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Refer to the lecture slides referenced in the handout to see how this is done.

        Remember that the loss is similar to the loss as in part 1, with a few specific changes.

        1) In your actor loss, instead of element-wise multiplying with discounted_rewards, you want to element-wise multiply with your advantage. 
        See handout/slides for definition of advantage.
        
        2) In your actor loss, you must use tf.stop_gradient on the advantage to stop the loss calculated on the actor network 
        from propagating back to the critic network.
        
        3) See handout/slides for how to calculate the loss for your critic network.

        :param states: A batch of states of shape (episode_length, state_size)
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a TensorFlow scalar
        """
        # TODO: implement this :)
        # Hint: use tf.gather_nd (https://www.tensorflow.org/api_docs/python/tf/gather_nd) to get the probabilities of the actions taken by the model
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        discounted_rewards = tf.convert_to_tensor(discounted_rewards)
        episode_length = tf.size(actions)
        
        values = self.value_function(states)
        values = tf.squeeze(values)
        advantage = tf.math.subtract(discounted_rewards, values)
        loss_critic = tf.math.multiply(advantage, advantage)
        loss_critic = tf.math.reduce_sum(loss_critic)
        
        tf.stop_gradient(advantage)
        probs = self.call(states)
        actions = tf.reshape(actions, [episode_length, 1])
        indices = tf.range(episode_length)
        indices = tf.reshape(indices, [episode_length, 1])
        actions = tf.concat((indices, actions), 1)
        probs = tf.gather_nd(probs, actions)
        probs = tf.math.log(probs)
        probs = tf.math.multiply(probs, advantage)
        loss_actor = -1.0 * tf.math.reduce_sum(probs)
        return (tf.math.add(loss_actor, loss_critic))