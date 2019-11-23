clear;clc;close all
%Note: this currently applies to MNIST and CIFAR10 datasets.
warning off;

set(0,'DefaultFigureVisible','on')
adv_type_vec = {'orig'};
orig_class_vec = 0:9;
% random_seed_vec = {'1234','2345','3456','4567','5678'};
random_seed_vec = {'1234'};

class_num = 10;
img_num = 100;
query_sum_collect = zeros(class_num,length(adv_type_vec));
% attack_method_vec = {'nes','autozoom'};
attack_method_vec = {'autozoom'};
dataset_vec = {'imagenet'};
prefix = '../';
model_vec = {'imagenet'};

max_query_num = 100000;
cost_base = 0;
%main program
for sss = 1:length(dataset_vec)
    dataset = dataset_vec{sss};
for s = 1:length(attack_method_vec)
    attack_method = attack_method_vec{s};
    mnist_cifar10_query = cell(1,2); % to plot the query distribution of mnist and cifar10, show query variance
    for i = 1:length(model_vec)
        model = model_vec{i};
        if strcmp(model,'madry_robust')
            target_type = 'untargeted';
        else
            target_type = 'targeted';
        end

        if strcmp(dataset,'mnist')
            local_type_vec = {'simple_local'};
        elseif strcmp(dataset,'cifar10') 
            local_type_vec = {'modelB_modelD_modelE','adv_densenet_adv_resnet'};
        else
            local_type_vec = {'normal_model'};
        end
        
        for kk = 1:length(local_type_vec)
            fprintf('******************************************************\n')
            local_type = local_type_vec{kk};
            opt_cell = cell(1,length(random_seed_vec));
            greedy_cell = cell(1,length(random_seed_vec));
            random_cell = cell(1,length(random_seed_vec));
            query_budget = 0;
            for ii =1:length(random_seed_vec)
                rand_seed = random_seed_vec{ii};
                violation_check = cell(1,2);
                for k = 1:length(adv_type_vec)
                    adv_type = adv_type_vec{k};
                    fprintf('--------------Printing %s %s %s %s %s %s Round %d :-----------------\n', dataset, attack_method,target_type,model,local_type,adv_type,ii);
                    bbox_file_path_head = [prefix '/' attack_method '/' 'Results/' rand_seed '/'];  
                    %% load black-box attack results
                    % load query numbers
                    query_num_name = [adv_type '_num_queries.txt'];
                    file_full_path=fullfile(bbox_file_path_head,query_num_name);
                    fileID = fopen(file_full_path);
                    formatSpec = '%f';
                    query_num_vec = fscanf(fileID,formatSpec);
                     if strcmp(adv_type,'orig') && strcmp(attack_method,'autozoom')
                         query_num_vec = query_num_vec + 1;
                     end
                     query_num_vec(query_num_vec == 0) = 1;
                     query_num_vec(query_num_vec > max_query_num) = max_query_num;
                     
                     % choose the query budget as the largest of all runs
                     if sum(query_num_vec) > query_budget
                         query_budget = sum(query_num_vec) ;
                     end
                     
                    % success rate vec
                    succ_rate_name = [adv_type '_success_flags.txt'];
                    file_full_path=fullfile(bbox_file_path_head,succ_rate_name);
                    fileID = fopen(file_full_path);
                    formatSpec = '%f';
                    succ_rate_vec = fscanf(fileID,formatSpec);
                    succ_query_num_vec = query_num_vec(logical(succ_rate_vec));

                    % read original img loss
                    file_path_head = bbox_file_path_head; 

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
                    opt_cell{ii} = opt_adv_num_vec;

                    adv_cost = sum(query_num_vec)/ sum(succ_rate_vec);
                    rand_adv_num = round((1/adv_cost)*(0:max_query_num*length(query_num_vec)));
                    rand_adv_num = round(rand_adv_num);
                    rand_adv_num(rand_adv_num>max(opt_adv_num_vec)) = max(opt_adv_num_vec);
                    all_adv_num_cell{end} = rand_adv_num;
                    random_cell{ii} = rand_adv_num;

                    % generate greedy adv num vector 
                    pgd_cnt_vec = 0;
                    greedy_init_adv_num = 0*(1:img_num);
                    [target_sorted_order, ~] = select_seed(pgd_cnt_vec,orig_img_loss,'img-loss');
                    [loss_greedy_adv_num_vec,~] = generate_adv_num_vec(query_num_vec,target_sorted_order,succ_rate_vec,0,max_query_num);
                    % two stage strategy with local + target info
                    all_adv_num_cell{2} = cat(1,greedy_init_adv_num',loss_greedy_adv_num_vec+greedy_init_adv_num(end)); 
                    greedy_cell{ii} = cat(1,greedy_init_adv_num',loss_greedy_adv_num_vec+greedy_init_adv_num(end)); 
                end
            end
            opt_tmp = 0;
            greedy_tmp = 0;
            random_tmp = 0;
            % collect quantitative info of opt strategy
            if strcmp(model,'madry_robust')
                fraction  = [0.01,0.02,0.04];
            else
                fraction  = [0.01,0.02,0.05,0.1];
            end
            opt_quanti_info_mat = zeros(length(random_seed_vec),length(fraction)+1);
            greedy_quanti_info_mat = zeros(length(random_seed_vec),length(fraction)+1);
            random_quanti_info_mat = zeros(length(random_seed_vec),length(fraction)+1);
            for ii = 1:length(random_seed_vec)
                opt_tmp = opt_tmp + opt_cell{ii};
                greedy_tmp = greedy_tmp + greedy_cell{ii};
                random_tmp = random_tmp + random_cell{ii};
                for haha = 1:length(fraction)
                    opt_quanti_info_mat(ii,haha+1) = find(opt_cell{ii} == fraction(haha)*img_num,1);
                    greedy_quanti_info_mat(ii,haha+1)  = find(greedy_cell{ii} == fraction(haha)*img_num,1);
                    random_quanti_info_mat(ii,haha+1)  = find(random_cell{ii} == fraction(haha)*img_num,1);
                end
                opt_quanti_info_mat(ii,1) = find(opt_cell{ii} == 1,1);
                greedy_quanti_info_mat(ii,1)  = find(greedy_cell{ii} == 1,1);
                random_quanti_info_mat(ii,1)  = find(random_cell{ii} == 1,1);
            end
            all_adv_num_cell{1} = round(opt_tmp/length(random_seed_vec));
            all_adv_num_cell{2} = round(greedy_tmp/length(random_seed_vec));
            all_adv_num_cell{3} = round(random_tmp/length(random_seed_vec));
            
            fprintf('Now printing statistical information\n');
            for ii = 1:length(fraction)+1
                if ii == 1
                    fprintf('First AE: %2f\n',fraction(ii))
                    fprintf('Opt: Mean (%2f) Std (%2f); Greedy: Mean (%2f) Std (%2f); Random: Mean (%2f) Std (%2f) \n',...
                        mean(opt_quanti_info_mat(:,1)),std(opt_quanti_info_mat(:,1)),...
                        mean(greedy_quanti_info_mat(:,1)),std(greedy_quanti_info_mat(:,1)),...
                        mean(random_quanti_info_mat(:,1)),std(random_quanti_info_mat(:,1)));
                else
                    fprintf('%f of seeds:\n',fraction(ii-1))
                    fprintf('Opt: Mean (%2f) Std (%2f); Greedy: Mean (%2f) Std (%2f); Random: Mean (%2f) Std (%2f) \n',...
                        mean(opt_quanti_info_mat(:,ii)),std(opt_quanti_info_mat(:,ii)),...
                        mean(greedy_quanti_info_mat(:,ii)),std(greedy_quanti_info_mat(:,ii)),...
                        mean(random_quanti_info_mat(:,ii)),std(random_quanti_info_mat(:,ii)));
                end
            end
                        
            % plot the graph containing optimal, random and different greedy strategies...
            legend_info = {'Retroactive Optimal','Target Loss Value','Random Schedule'};
            %marker_info = {'--k*','-gs','-rd'};
            marker_info = {'-*','-s','-d'};
            fprintf('Now printing ALL scheduling strategies\n');
            plot_graph_greedy_explore(query_budget,all_adv_num_cell,legend_info,marker_info,round(query_budget/4));
        end
    end
end
end
