clear;clc;close all
%Note: this currently applies to MNIST and CIFAR10 datasets.
warning off;

set(0,'DefaultFigureVisible','on')

%model_vec = {'imagenet','mnist_robust','mnist', 'cifar10','cifar10_robust',};
model_vec = {'madry_robust','densenet'};%'mnist','mnist_robust', 'cifar10'};
adv_type_vec = {'no_tune'};
%adv_type_vec = {'no_tune'};

orig_class_vec = 0:9;
class_num = 10;
img_num = 1000;
query_sum_collect = zeros(class_num,length(adv_type_vec));
prefix = '..';
attack_method_vec = {'autozoom','nes',};
%attack_method_vec = {'autozoom'};
dist_metric = 'li';
loss_func = 'cw';

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
        if strcmp(model,'imagenet')
            max_query_num = 100000;
        end
        % check if local models are needed 
        if strcmp(model,'mnist') || strcmp(model,'mnist_robust') || strcmp(model,'imagenet')
             %tar_class_vec = [2,9,3,5,5,3,5,9,3,4];
             local_type_vec = {'simple_local'};
        elseif strcmp(model,'densenet') || strcmp(model,'madry_robust')
            %tar_class_vec = [6,2,8,0,0,0,7,8,4,4];
             local_type_vec = {'adv_densenet_adv_resnet','modelB_modelD_modelE'};
        end
        
        for kk = 1:length(local_type_vec)
            fprintf('******************************************************\n')
            local_type = local_type_vec{kk};
            violation_check = cell(1,2);
            for k = 1:length(adv_type_vec)
                adv_type = adv_type_vec{k};
                if strcmp(adv_type,'norm')
                    local_type = 'no_local_model';
                end
                % fprintf('--------------Printing %s %s %s %s %s:-----------------\n', attack_method,target_type,model,local_type,adv_type);
             %% Load related files
                % load query number and success rate vec
                if (strcmp(model,'densenet') || strcmp(model,'madry_robust'))&&strcmp(adv_type,'norm')
                    local_type = 'no_local_model';
                elseif strcmp(model,'mnist') || strcmp(model,'mnist_robust')
                    %file_path_head1 = [prefix '/' attack_method '/' model '/' target_type '/'];
                    local_type = 'no_local_model';
                end
                fprintf('--------------Printing %s %s %s %s %s:-----------------\n', attack_method,target_type,model,local_type,adv_type);
                file_path_head1 = [prefix '/' attack_method '/' model '/' target_type '/' local_type '/' loss_func '/'];
                if strcmp(model,'densenet') || strcmp(model,'madry_robust')
                    c_val_str = '_cval0.05_';
                elseif strcmp(model,'mnist') || strcmp(model,'mnist_robust')
                    c_val_str = '_cval0.3_';
                end
                query_num_name = ['query_num_vec_all' c_val_str adv_type '_random.txt'];
                file_full_path=fullfile(file_path_head1,query_num_name);
                fileID = fopen(file_full_path);
                formatSpec = '%f';
                query_num_vec = fscanf(fileID,formatSpec);
                if min(query_num_vec) == 0
                    query_num_vec = query_num_vec + 1;
                elseif strcmp(attack_method,'autozoom') && strcmp(adv_type,'norm')
                    query_num_vec = query_num_vec + 1;
                end
                if max(query_num_vec) > 4000
                    query_num_vec(query_num_vec>4000) = 4000; 
                end

                % success rate vec
                succ_rate_name = ['succ_vec_all' c_val_str adv_type '_random.txt'];
                file_full_path=fullfile(file_path_head1,succ_rate_name);
                fileID = fopen(file_full_path);
                formatSpec = '%f';
                succ_rate_vec = fscanf(fileID,formatSpec);
                succ_query_num_vec = query_num_vec(logical(succ_rate_vec));
                % load pgd cnt vectors
                file_path_head = [prefix '/sub_saved_qual/' model '/' target_type '/' local_type '/' loss_func '/'];
                pgd_cnt_name = ['pgd_cnt_mat.txt'];
                file_full_path=fullfile(file_path_head,pgd_cnt_name);
                fileID = fopen(file_full_path);
                formatSpec = '%f %f %f';
                pgd_cnt_vec = fscanf(fileID,formatSpec);
                pgd_cnt_vec = reshape(pgd_cnt_vec,[length(pgd_cnt_vec)/length(query_num_vec),length(query_num_vec)]);
                pgd_cnt_vec = pgd_cnt_vec';
                % load adv image loss vectors
                adv_loss_name = ['adv_img_loss.txt'];
                file_full_path=fullfile(file_path_head,adv_loss_name);
                fileID = fopen(file_full_path);
                formatSpec = '%f';
                adv_img_loss = fscanf(fileID,formatSpec);
                % read original img loss
                orig_loss_name = ['orig_img_loss.txt'];
                file_full_path=fullfile(file_path_head,orig_loss_name);
                fileID = fopen(file_full_path);
                formatSpec = '%f';
                orig_img_loss = fscanf(fileID,formatSpec);
                
                orig_loss_name = ['succ_model_num.txt'];
                file_full_path=fullfile(file_path_head,orig_loss_name);
                fileID = fopen(file_full_path);
                formatSpec = '%f';
                succ_model_num = fscanf(fileID,formatSpec);
                
                file_name = 'max_gap.txt';
                file_full_path=fullfile(file_path_head,file_name);
                fileID = fopen(file_full_path);
                formatSpec = '%f';
                thres_vec_max_gap = fscanf(fileID,formatSpec);  

                file_name = 'min_gap.txt';
                file_full_path=fullfile(file_path_head,file_name);
                fileID = fopen(file_full_path);
                formatSpec = '%f';
                thres_vec_min_gap = fscanf(fileID,formatSpec);  

                file_name = 'ave_gap.txt';
                file_full_path=fullfile(file_path_head,file_name);
                fileID = fopen(file_full_path);
                formatSpec = '%f';
                thres_vec_ave_gap = fscanf(fileID,formatSpec);  
             %% end of loading files
             
                % generate optimal adv num vector 
                two_stage_num = 4; % I set it to 4 to count for random ordering
                %metrics = {'PGD-Step','Max-Gap','Min-Gap','Ave-Gap'};
                metrics = {'PGD-Step'};
                trans_rate_vec = adv_img_loss == 0;    
                trans_rate = sum(trans_rate_vec)/length(trans_rate_vec);
                all_adv_num_cell = cell(1,two_stage_num); % first is the strategy directly applies local info based order
                for sss = 1:length(metrics)
                    metric = metrics{sss};
                    if strcmp(metric,'PGD-Step')
                        [local_sorted_order, candi_rate_vec] = select_seed(pgd_cnt_vec,-thres_vec_ave_gap,'mixture');
                    else
                        if strcmp(metric,'Max-Gap')
                            metric_value = thres_vec_max_gap;
                            sort_direction = 'descend';
                        elseif strcmp(metric,'Min-Gap')
                            metric_value = thres_vec_min_gap;
                            sort_direction = 'descend';
                        elseif strcmp(metric,'Ave-Gap')
                            metric_value = thres_vec_ave_gap;
                            sort_direction = 'descend';
                        end
                        [tmp_thres_vec,tmp_idx_] = sort(metric_value,sort_direction);
                        % now first group by number of local models
                        % attacked and then sort by specific metrics
                        local_sorted_order = [];
                        tmp_succ_model_num = succ_model_num(tmp_idx_);
                        for hh =max(tmp_succ_model_num):-1:min(tmp_succ_model_num)
                            tmp_group_idx = tmp_succ_model_num ==  hh;
                            tmp_group_idx = tmp_idx_(tmp_group_idx);
                            local_sorted_order = cat(1,local_sorted_order,tmp_group_idx);
                        end
                    end
                    [no_two_stage_adv_num_vec,~] = generate_adv_num_vec(query_num_vec,local_sorted_order,succ_rate_vec,0,max_query_num);
                    all_adv_num_cell{4} = no_two_stage_adv_num_vec;

                    % get the initial direct transfers
                    trans_vec_greedy_sorted = trans_rate_vec(local_sorted_order);
                    greedy_init_adv_num = zeros(1,img_num);
                    tmp_cnt = 0;
                    for pgd_ite = 1:img_num
                        if trans_vec_greedy_sorted(pgd_ite) > 0
                            tmp_cnt = tmp_cnt + 1;
                        end
                        greedy_init_adv_num(pgd_ite) = tmp_cnt;
                    end
                    [target_sorted_order, ~] = select_seed(pgd_cnt_vec,adv_img_loss,'img-loss');
                    trans_rate_vec_target_sorted = trans_rate_vec(target_sorted_order);
                    sorted_idx_img_loss = target_sorted_order(~trans_rate_vec_target_sorted);
                    [loss_greedy_adv_num_vec,~] = generate_adv_num_vec(query_num_vec,sorted_idx_img_loss,succ_rate_vec,0,max_query_num);
                    % two stage strategy with local + target info
                    all_adv_num_cell{1} = cat(1,greedy_init_adv_num',loss_greedy_adv_num_vec+greedy_init_adv_num(end)); 

                    % two stage strategy with only local info
                    trans_rate_vec_local_sorted = trans_rate_vec(local_sorted_order);
                    sorted_idx_local_info = local_sorted_order(~trans_rate_vec_local_sorted);
                    [local_greedy_adv_num_vec,~] = generate_adv_num_vec(query_num_vec,sorted_idx_local_info,succ_rate_vec,0,max_query_num);
                    % two stage strategy with local + target info
                    all_adv_num_cell{2} = cat(1,greedy_init_adv_num',local_greedy_adv_num_vec+greedy_init_adv_num(end)); 
                    % add random ordering to all_adv_num_cell 
                    
                    query_budget = sum(query_num_vec);
                    % plot the graph containing optimal, random and different
                    % greedy strategies...
                    legend_info = {'Prioritize on Target Info','Prioritize on Local Info','Random Order'};
                    marker_info = {'-g*','-bs','-rd'};
                    color_opt_vec = {'g','b','r'};
                    fprintf('Now printing ALL scheduling strategies of Metric: %s\n',metric);
                    %plot_graph_greedy_explore(query_budget,all_adv_num_cell,legend_info,marker_info,round(query_budget/4));            
                    
                    % lot the random ordering case 
                    non_trans_query_num_vec = query_num_vec(~trans_rate_vec);
                    K = 50;
                    rand_adv_num = 0;
                    for rnd_ite = 1:K
                        sorted_idx = randperm(length(non_trans_query_num_vec));              
                        [orig_rnd_adv_num_tmp,~] = generate_adv_num_vec(non_trans_query_num_vec,sorted_idx,succ_rate_vec(~trans_rate_vec),0,max_query_num);
                        rand_adv_num = rand_adv_num + orig_rnd_adv_num_tmp;
                    end
                    rand_adv_num = rand_adv_num/K;
                    rand_adv_num = round(rand_adv_num);
                    all_adv_num_cell{3} = cat(1,greedy_init_adv_num',rand_adv_num+greedy_init_adv_num(end)); 
                
                    % plot the greedy adv number plot
                    order_value = floor(log10(query_budget));
                    figure;
                    marker_index = round(query_budget/4);
                    for ss = 1:length(all_adv_num_cell)-1 % do not include the stageless strategy here
                        adv_num = all_adv_num_cell{ss};
                        color_opt = color_opt_vec{ss};
                        plot(img_num+1:query_budget,adv_num(img_num+2:query_budget+1),marker_info{ss},'MarkerSize',12,'MarkerIndices', [marker_index],...
                        'LineWidth',2);
                        xlim([img_num+1  query_budget]);
                        ylim([0 Inf]);
%                         xticks([img_num+1 round(query_budget/(4)) round(2*query_budget/(4)) ...
%                               round(3*query_budget/(4)) round(query_budget)])

%                          xticks([img_num+1 round(query_budget/(4*order_value),1) round(2*query_budget/(4*order_value),1) ...
%                              round(3*query_budget/(4*order_value),1) round(query_budget/order_value,1)])
%                          xticks([img_num+1 round(query_budget/(10),-(order_value-1)) round(2*query_budget/(10),-(order_value-1)) ...
%                              round(3*query_budget/(10),-(order_value-1)) round(4*query_budget/(10),-(order_value-1)) round(query_budget,-(order_value-1))])
                        hold on;
                        if query_budget <= 1000
                            xticks([0 100 200 300 400 500 600 700 800 900 1000])
                            xticklabels({'0' '100' '200' '300' '400' '500' '600' '700' '800' '900' '1000'})
                        else
                            xticks([img_num+1  3e5 6e5 9e5 12e5 ])
                            yticks([0 100 200 300 400 500 600 max(adv_num(img_num+2:query_budget+1))])
                        end
                        
                    end
                    %legend(legend_info,'FontSize', 18);
                    xlabel('Budget on Number of Queries','FontSize', 18);
                    ylabel('# of AEs Found','FontSize', 18);
                                        
                    % print relevant info...
                    fprintf('Now printing statistical information of Metric: %s\n',metric);
                    fraction  = [trans_rate+0.01,trans_rate+0.02,trans_rate+0.05,trans_rate+0.1, trans_rate+0.2];
                    fraction_vec_cell = cell(1,length(all_adv_num_cell));
                    for hehe = 1:length(all_adv_num_cell)
                        fraction_query_vec = zeros(1,length(fraction)+1);
                        data = all_adv_num_cell{hehe};
                        for haha = 1:length(fraction)
                            fraction_query_vec(haha) = find(data == floor(fraction(haha)*img_num),1)-img_num;
                        end
                        fraction_query_vec(haha+1) = find(data==1,1);
                        fraction_vec_cell{hehe} = fraction_query_vec;
                        disp(fraction_query_vec)
                    end

                    query_budget = img_num;
                    first_stage_cell = cell(1,3);
                    first_stage_cell{1} = all_adv_num_cell{3};
                    first_stage_cell{2} = floor((0:img_num)*(sum(trans_rate_vec)/img_num)); 
                    first_stage_cell{3} = all_adv_num_cell{4};
                    
                    %plot the firts stages...
                    figure;
                    marker_index = round(query_budget/4);
                    legend_info =  {'Transfer (PGD-Step)','Transfer (Random)'};
                    colot_opt_vec = {'b','r'};
                    for ss = 1:length(first_stage_cell)-1 % do not include the stageless strategy here
                        adv_num = first_stage_cell{ss};
                        color_oot = color_opt_vec{ss};
                        plot(0:query_budget,adv_num(1:query_budget+1),marker_info{ss+1},'MarkerSize',12,'MarkerIndices', [marker_index],...
                        'LineWidth',2);
                        xlim([1  query_budget]);
                        ylim([0 Inf]);

%                          xticks([img_num+1 round(query_budget/(4*order_value),1) round(2*query_budget/(4*order_value),1) ...
%                              round(3*query_budget/(4*order_value),1) round(query_budget/order_value,1)])
%                          xticks([img_num+1 round(query_budget/(10),-(order_value-1)) round(2*query_budget/(10),-(order_value-1)) ...
%                              round(3*query_budget/(10),-(order_value-1)) round(4*query_budget/(10),-(order_value-1)) round(query_budget,-(order_value-1))])
                        hold on;
                        if query_budget <= 1000
                            xticks([0 100 200 300 400 500 600 700 800 900 1000])
                            xticklabels({'0' '100' '200' '300' '400' '500' '600' '700' '800' '900' '1000'})
                        end
                    end
                    %legend(legend_info,'FontSize', 18);
                    xlabel('Budget on Number of Queries','FontSize', 18);
                    ylabel('# of AEs Found','FontSize', 18);
                    
                    
                    % print relevant info of first stage...
                    fprintf('Now printing statistical information of Metric: %s\n',metric);
                    fraction  = [0.01,0.02,0.05];
                    fraction_vec_cell = cell(1,length(first_stage_cell));
                    for hehe = 1:length(first_stage_cell)
                        fraction_query_vec = zeros(1,length(fraction)+1);
                        data = first_stage_cell{hehe};
                        for haha = 1:length(fraction)
                            fraction_query_vec(haha) = find(data == fraction(haha)*img_num,1);
                        end
                        fraction_query_vec(haha+1) = find(data==1,1);
                        fraction_vec_cell{hehe} = fraction_query_vec;
                        disp(fraction_query_vec)
                    end

%                     % plot the graph containing optimal, random and different
%                     % greedy strategies...
%                     legend_info = {'Transfer Check (PGD-Step)','Transfer Check (Random Order)','Exhaustive Execution'};
%                     fprintf('Now printing Zoomed scheduling strategies of Metric: %s \n',metric);
%                     plot_graph_greedy_explore(query_budget,all_adv_num_cell(3:4),legend_info,marker_info,query_budget/2);
                end
            end
        end
    end
end
