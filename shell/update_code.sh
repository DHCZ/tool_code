#!/bin/sh
sudo apt-get install tmux
session="update_code"

# set up tmux
tmux start-server

tmux new-session -d -s $session

# Shaanqi-007
tmux selectp -t 0
tmux send-keys "ssh tusimple@10.132.132.202" C-m
tmux splitw -h -p  50

# Shaanqi-008
tmux selectp -t 1
tmux send-keys "ssh tusimple@10.132.135.235" C-m
tmux splitw -v -p 50

# Shaanqi-0010
tmux selectp -t 1
tmux splitw -v -p 50
tmux selectp -t 2
tmux send-keys "ssh tusimple@10.132.135.216" C-m

# Shaanqi-013
tmux selectp -t 3
tmux send-keys "ssh tusimple@10.132.131.252" C-m
tmux splitw -v -p 50

# Shaanqi-014
tmux selectp -t 4
tmux send-keys "ssh tusimple@10.132.135.108" C-m

tmux selectp -t 0
tmux setw synchronize-pane
# Finished setup, stay simple!
tmux attach-session -t $session
