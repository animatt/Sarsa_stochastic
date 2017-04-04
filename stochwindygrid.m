clear, clc, close all

% This is an implementation of Sarsa that finds the shortest mean path
% between too specific locations on a given gridworld with a stochastic
% wind blowing across it.

% initialize world
world = ones(20, 30) * diag(randi([-1 1], 30, 1));
[m, n] = size(world);

start = {3 2};
finish = [17 27];

% initialize agent (row, col[, rowstep, colstep])
Q = zeros(m, n, 3, 3);
policy = ones(m, n);
alpha = 0.2;

count = 0;

converging = true;
while converging
    
    [row, col] = start{:};
    action = datasample([randi(9) policy(row, col)], 1, 'Weights', [0.1 0.9]);
    [rstep, cstep] = ind2sub([3 3], action);
    
    episode = [row col rstep cstep]';
    ep = [zeros(4, 1) episode];
    
    seeking_end_position = true;
    while seeking_end_position
        
        % update position
        row = bound(row + world(row, col) + rstep + randi([-1 1]) - 2, m);
        col = bound(col + cstep - 2, n);
        
        % take an action
        action = datasample([randi(9) policy(row, col)], 1, 'Weights', [.1 .9]);
        [rstep, cstep] = ind2sub([3 3], action);
        
        episode = [row col rstep cstep]';
        ep = [ep(:, 2) episode];
        
        % check whether goal was met
        if isequal([row col], finish)
            seeking_end_position = false;
        end
        
        % evaluate policy
        SA = sub2ind(size(Q), ep(1, 1), ep(2, 1), ep(3, 1), ep(4, 1));
        SAnext = sub2ind(size(Q), row, col, rstep, cstep);
        Q(SA) = Q(SA) + alpha * (-1 + Q(SAnext) - Q(SA));
        
        % improve policy
        [~, action] = max(Q(ep(1, 1), ep(2, 1), :), [], 3);
        policy(ep(1, 1), ep(2, 1)) = action;
    end
    
    count = count + 1;
end