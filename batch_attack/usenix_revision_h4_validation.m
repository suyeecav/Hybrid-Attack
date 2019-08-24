clear;clc;close all
%Note: this currently applies to MNIST and CIFAR10 datasets.
warning off;

set(0,'DefaultFigureVisible','on')

model_vec = {'imagenet','madry_robust','mnist_robust','mnist', 'densenet',};
% model_vec = {'mnist','mnist_robust'};
% model_vec = {'madry_robust','densenet'};
adv_type_vec = {'norm'};
%adv_type_vec = {'no_tune'};
loss_func = 'cw';
orig_class_vec = 0:9;
class_num = 10;
img_num = 1000;
query_sum_collect = zeros(class_num,length(adv_type_vec));
prefix = '..';
%attack_method_vec = {'nes_bandit','nes','autozoom'};
attack_method_vec = {'nes','autozoom',};
dist_metric = 'li';

max_query_num = 4000;
cost_base = 0;
%main program
for s = 1:length(attack_method_vec)
    attack_method = attack_method_vec{s};
    mnist_cifar10_query = cell(1,2); % to plot the query distribution of mnist and cifar10, show query variance
    for i = 1:length(model_vec)
        model = model_vec{i};
        if strcmp(model,'mnist_robust') || strcmp(model,'madry_robust')
            target_type = 'untargeted';
        else
            target_type = 'targeted';
        end
        if strcmp(model,'imagenet') || strcmp(model,'imagenet_bandit')
            max_query_num = 100000;
        end
        % check if local models are needed 
        if strcmp(model,'mnist') || strcmp(model,'mnist_robust') 
             %tar_class_vec = [2,9,3,5,5,3,5,9,3,4];
             local_type_vec = {'simple_local'};
        elseif strcmp(model,'densenet') || strcmp(model,'madry_robust')
            %tar_class_vec = [6,2,8,0,0,0,7,8,4,4];
             % local_type_vec = {'adv_densenet_adv_resnet','modelB_modelD_modelE'};
             local_type_vec = {'no_local_model'};
        elseif strcmp(model,'imagenet') || strcmp(model,'imagenet_bandit')
             local_type_vec = {'repeated'};
        end
        
        for kk = 1:length(local_type_vec)
            fprintf('******************************************************\n')
            local_type = local_type_vec{kk};
            violation_check = cell(1,2);
            for k = 1:length(adv_type_vec)
                adv_type = adv_type_vec{k};
%                 if strcmp(adv_type,'norm')
%                     local_type = 'no_local_model';
%                 end
                % fprintf('--------------Printing %s %s %s %s %s:-----------------\n', attack_method,target_type,model,local_type,adv_type);
             %% Load related files
                % load query number and success rate vec
                if (strcmp(model,'densenet') || strcmp(model,'madry_robust'))&&strcmp(adv_type,'norm')
                    local_type = 'no_local_model';
                elseif strcmp(model,'mnist') || strcmp(model,'mnist_robust')
                    %file_path_head1 = [prefix '/' attack_method '/' model '/' target_type '/'];
                    local_type = 'simple_local';
                end
                fprintf('--------------Printing %s %s %s %s %s:-----------------\n', attack_method,target_type,model,local_type,adv_type);
                file_path_head1 = [prefix '/' attack_method '/' model '/' target_type '/' local_type '/' loss_func '/'];
                if strcmp(model,'densenet') || strcmp(model,'madry_robust')
                    c_val_str = '_cval0.05_';
                elseif strcmp(model,'mnist') || strcmp(model,'mnist_robust')
                    c_val_str = '_cval0.3_';
                elseif  strcmp(model,'imagenet') ||  strcmp(model,'imagenet_bandit')
                    c_val_str = '_cval12.0_';
                end
                query_num_name = ['query_num_vec_all' c_val_str adv_type '_random.txt'];
                file_full_path=fullfile(file_path_head1,query_num_name);
                fileID = fopen(file_full_path);
                formatSpec = '%f';
                query_num_vec = fscanf(fileID,formatSpec);
                if strcmp(adv_type,'norm') && strcmp(attack_method,'autozoom')
                    query_num_vec = query_num_vec + 1;
                end

                % success rate vec
                succ_rate_name = ['succ_vec_all' c_val_str adv_type '_random.txt'];
                file_full_path=fullfile(file_path_head1,succ_rate_name);
                fileID = fopen(file_full_path);
                formatSpec = '%f';
                succ_rate_vec = fscanf(fileID,formatSpec);
                succ_query_num_vec = query_num_vec(logical(succ_rate_vec));
                pgd_cnt_vec = 0;
                % load the original image loss
                file_path_head = [prefix '/sub_saved_qual/' model '/' target_type '/' local_type '/' loss_func '/'];
                % read original img loss
                orig_loss_name = ['orig_img_loss.txt'];
                file_full_path=fullfile(file_path_head,orig_loss_name);
                fileID = fopen(file_full_path);
                formatSpec = '%f';
                orig_img_loss = fscanf(fileID,formatSpec);
             %% end of loading files
                % generate optimal adv num vector 
                all_metrics = {'retroactive optimal','target loss','random',};
                all_adv_num_cell = cell(1,length(all_metrics)); % first and second are optimal and random vector
                [~,sorted_order] = sort(query_num_vec);
                [opt_adv_num_vec,~] = generate_adv_num_vec(query_num_vec,sorted_order,succ_rate_vec,0,max_query_num);
                all_adv_num_cell{1} = opt_adv_num_vec;
                % generate random adv num vector
                %{
                K = 50;
                rand_adv_num = 0;
                for rnd_ite = 1:K
                    sorted_idx = randperm(length(query_num_vec));              
                    [orig_rnd_adv_num_tmp,~] = generate_adv_num_vec(query_num_vec,sorted_idx,succ_rate_vec,0,max_query_num);
                    rand_adv_num = rand_adv_num + orig_rnd_adv_num_tmp;
                end
                rand_adv_num = rand_adv_num/K;
                rand_adv_num = round(rand_adv_num);
                %}
                adv_cost = sum(query_num_vec)/ sum(succ_rate_vec);
                rand_adv_num = round((1/adv_cost)*(0:sum(query_num_vec)));
                rand_adv_num = round(rand_adv_num);
                all_adv_num_cell{end} = rand_adv_num;
                
                % generate greedy adv num vector 
                greedy_init_adv_num = 0*(1:img_num);
                [target_sorted_order, ~] = select_seed(pgd_cnt_vec,orig_img_loss,'img-loss');
                [loss_greedy_adv_num_vec,~] = generate_adv_num_vec(query_num_vec,target_sorted_order,succ_rate_vec,0,max_query_num);
                % two stage strategy with local + target info
                all_adv_num_cell{2} = cat(1,greedy_init_adv_num',loss_greedy_adv_num_vec+greedy_init_adv_num(end)); 

                query_budget = sum(query_num_vec);
                % plot the graph containing optimal, random and different
                % greedy strategies...
                legend_info = {'Retroactive Optimal','Target Loss Value','Random Schedule'};
                marker_info = {'-*','-s','-d'};
                fprintf('Now printing ALL scheduling strategies\n');
                plot_graph_greedy_explore(query_budget,all_adv_num_cell,legend_info,marker_info,round(query_budget/4));

                 % print relevant info...
                 fprintf('Now printing statistical information \n');
                 if sum(succ_rate_vec)/length(succ_rate_vec)<0.1
                     fraction  = [0.01,0.02];
                 else
                    fraction  = [0.01,0.02,0.05,0.1];
                 end
                 fraction_vec_cell = cell(1,length(all_adv_num_cell));
                 for hehe = 1:length(all_adv_num_cell)
                     fraction_query_vec = zeros(1,length(fraction)+1);
                     data = all_adv_num_cell{hehe};
                     for haha = 1:length(fraction)
                         fraction_query_vec(haha) = find(data == fraction(haha)*img_num,1);
                     end
                     fraction_query_vec(haha+1) = find(data==1,1);
                     fraction_vec_cell{hehe} = fraction_query_vec;
                     disp(fraction_query_vec)
                 end
            end
        end
    end
end
