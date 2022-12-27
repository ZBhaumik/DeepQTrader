#TRAIN
import logging
import coloredlogs
from docopt import docopt

def main(train_stock, val_stock, window_size, batch_size, ep_count,
         strategy="t-dqn", model_name="model_debug", pretrained=False,
         debug=False, init_episode = 1):
    agent = Agent(window_size, strategy=strategy, pretrained=pretrained, model_name=model_name)
    
    train_data = yfinance_retrieve(train_stock, 0)
    val_data = yfinance_retrieve(val_stock, 1)

    initial_offset = val_data[1] - val_data[0]

    for episode in range(init_episode, ep_count + 1):
        train_result = train_model(agent, episode, train_data, ep_count=ep_count,
                                   batch_size=batch_size, window_size=window_size)
        val_result, _ = evaluate_model(agent, val_data, window_size, debug)
        show_train_result(train_result, val_result, initial_offset)


coloredlogs.install(level="DEBUG")
switch_k_backend_device()
main("MSFT","MSFT", window_size=10, batch_size=32, ep_count=10, model_name="model_debug_5", pretrained=True, init_episode=5)