import os
import sys
import gym
from pylab import *
import numpy as np
import tensorflow as tf
from reinforce import Reinforce
from reinforce_with_baseline import ReinforceWithBaseline


def visualize_data(total_rewards):
    """
    HELPER - do not edit.
    Takes in array of rewards from each episode, visualizes reward over episodes

    :param total_rewards: List of rewards from all episodes
    """

    x_values = arange(0, len(total_rewards), 1)
    y_values = total_rewards
    plot(x_values, y_values)
    xlabel('episodes')
    ylabel('cumulative rewards')
    title('Reward by Episode')
    grid(True)
    show()


def discount(rewards, discount_factor=.99):
    """
    Takes in a list of rewards for each timestep in an episode, and
    returns a list of the discounted rewards for each timestep.
    Refer to the slides to see how this is done.

    :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
    :param discount_factor: Gamma discounting factor to use, defaults to .99
    :returns: discounted_rewards: list containing the discounted rewards for each timestep in the original rewards list
    """
    # TODO: Compute discounted rewards
    rewards = tf.convert_to_tensor(rewards)
    timesteps = tf.size(rewards)
    discounted_rewards = []
    for t in range(timesteps):
        subrewards = rewards[t:]
        gamma = tf.range((timesteps-t), dtype=tf.float32)
        gamma = tf.math.pow(discount_factor, gamma)
        subrewards = tf.math.multiply(subrewards, gamma)
        discounted_rewards.append(tf.math.reduce_sum(subrewards))
    return tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)


def generate_trajectory(env, model):
    """
    Generates lists of states, actions, and rewards for one complete episode.

    :param env: The openai gym environment
    :param model: The model used to generate the actions
    :returns: A tuple of lists (states, actions, rewards), where each list has length equal to the number of timesteps in the episode
    """
    states = []
    actions = []
    rewards = []
    state = env.reset()
    done = False

    while not done:
        # TODO:
        # 1) use model to generate probability distribution over next actions
        # 2) sample from this distribution to pick the next action
        states.append(state)
        
        state = tf.reshape(state, [1, 4])
        probs = model.call(state)
        probs = np.array(probs)
        probs = np.squeeze(probs)
        #probs = probs / np.sum(probs)
        action = np.random.choice(model.num_actions, p=probs)
        
        actions.append(action)
        state, rwd, done, _ = env.step(action)
        rewards.append(rwd)

    return states, actions, rewards


def train(env, model):
    """
    This function should train your model for one episode.
    Each call to this function should generate a complete trajectory for one
    episode (lists of states, action_probs, and rewards seen/taken in the episode), and
    then train on that data to minimize your model loss.
    Make sure to return the total reward for the episode

    :param env: The openai gym environment
    :param model: The model
    :returns: The total reward for the episode
    """

    # TODO:
    # 1) Use generate trajectory to run an episode and get states, actions, and rewards.
    # 2) Compute discounted rewards.
    # 3) Compute the loss from the model and run backpropagation on the model.
    optimizer = tf.keras.optimizers.Adam(learning_rate=model.learning_rate)
    with tf.GradientTape() as tape:
        states, actions, rewards = generate_trajectory(env, model)
        states = tf.convert_to_tensor(states)
        actions = tf.convert_to_tensor(actions)
        rewards = tf.convert_to_tensor(rewards) #tf or np?
    
        discounted_rewards = discount(rewards)
        episode_loss = model.loss(states, actions, discounted_rewards)
    gradients = tape.gradient(episode_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return tf.math.reduce_sum(rewards)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"REINFORCE", "REINFORCE_BASELINE"}:
        print("USAGE: python assignment.py <Model Type>")
        print("<Model Type>: [REINFORCE/REINFORCE_BASELINE]")
        exit()

    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Initialize model
    if sys.argv[1] == "REINFORCE":
        model = Reinforce(state_size, num_actions) 
    elif sys.argv[1] == "REINFORCE_BASELINE":
        model = ReinforceWithBaseline(state_size, num_actions)

    # TODO: 
    # 1) Train your model for 650 episodes, passing in the environment and the agent. 
    # 2) Append the total reward of the episode into a list keeping track of all of the rewards. 
    # 3) After training, print the average of the last 50 rewards you've collected.
    rewards = []
    for i in range(650):
        rewards.append(train(env, model))
    print(tf.math.reduce_sum(rewards[-50:]) / 50.0)

    # TODO: Visualize your rewards.
    visualize_data(rewards)


if __name__ == '__main__':
    main()

