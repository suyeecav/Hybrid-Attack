%------------------------------------------------------------%
% this script is for prioritization in two-stage strategies %
%------------------------------------------------------------%

clear;clc;close all
%Note: this currently applies to MNIST and CIFAR10 datasets.
warning off;

set(0,'DefaultFigureVisible','on')

adv_type_vec = {'adv_no_tune'};
% random_seed_vec = {'1234','2345','3456','4567','5678'};
random_seed_vec = {'1234'};

orig_class_vec = 0:9;
class_num = 10;
img_num = 1000;
query_sum_collect = zeros(class_num,length(adv_type_vec));
% attack_method_vec = {'autozoom','nes'};
attack_method_vec = {'autozoom'};
metric = 'PGD-Step';
% dataset_vec = {'cifar10', 'mnist'};
dataset_vec = {'cifar10'};

max_query_num = 4000;
cost_base = 0;

%main program
for sss = 1:length(dataset_vec)
    dataset = dataset_vec{sss};
    if strcmp(dataset,'mnist')
        model_vec = {'madry_robust','mnist'};
        local_info_prefix = ['../' dataset '/local_info'];
        bbox_result_prefix =  ['../' dataset '/Results'];
    else
        % model_vec = {'madry_robust','densenet'};
        model_vec = {'madry_robust'};
        local_info_prefix = ['../' dataset '/local_info'];
        bbox_result_prefix =  ['../' dataset '/Results'];
    end
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
        end
        
        for kk = 1:length(local_type_vec)
            fprintf('******************************************************\n')
            local_type = local_type_vec{kk};
            local_greedy_cell = cell(1,length(random_seed_vec));
            target_greedy_cell = cell(1,length(random_seed_vec));
            random_cell = cell(1,length(random_seed_vec));
            trans_rate_cell = zeros(1,length(random_seed_vec));
            query_budget = 0;
            for ii = 1:length(random_seed_vec)
                rand_seed = random_seed_vec{ii};
                violation_check = cell(1,2);
                for k = 1:length(adv_type_vec)
                    adv_type = adv_type_vec{k};
                    fprintf('--------------Printing %s %s %s %s %s %s Round %d :-----------------\n', dataset, attack_method,target_type,model,local_type,adv_type,ii);
                    %bbox_file_path_head = [bbox_result_prefix '/' attack_method '/' model '/' target_type '/' rand_seed '/'];
                    if strcmp(dataset,'mnist')
                        bbox_file_path_head = [bbox_result_prefix '/' attack_method '/' model '/' target_type '/' rand_seed '/'];
                    else
                        bbox_file_path_head = [bbox_result_prefix '/' attack_method '/' model '/' target_type '/' local_type '/' rand_seed '/'];
                    end     
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
                    
                    %% load local mode information 
                    if strcmp(dataset,'mnist')
                        file_path_head = [local_info_prefix '/' model '/' target_type '/' rand_seed '/'];
                    else
                        file_path_head = [local_info_prefix '/' model '/' target_type '/' local_type '/' rand_seed '/'];
                    end 
                    % load pgd cnt vectors
                    % file_path_head = [local_info_prefix '/' model '/' target_type '/' rand_seed '/'];
                    pgd_cnt_name = ['pgd_cnt_mat.txt'];
                    file_full_path=fullfile(file_path_head,pgd_cnt_name);
                    fileID = fopen(file_full_path);
                    formatSpec = '%f %f %f';
                    pgd_cnt_vec = fscanf(fileID,formatSpec);
                    pgd_cnt_vec = reshape(pgd_cnt_vec,[length(pgd_cnt_vec)/length(query_num_vec),length(query_num_vec)]);
                    pgd_cnt_vec = pgd_cnt_vec';
                    
                    % from the pgd step numbers, infer number of successfully attacked local models
                    loc_model_num = size(pgd_cnt_vec,2);
                    failed_num = 1e10;
                    succ_model_num = zeros(size(pgd_cnt_vec,1),1);
                    for iii = 1:size(pgd_cnt_vec,1)
                        for j = 1:loc_model_num
                            if pgd_cnt_vec(iii,loc_model_num-j+1) < failed_num
                                succ_model_num(iii) = loc_model_num-j+1;
                                break;
                            end
                        end
                    end
                    
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
                    thres_vec_ave_gap = adv_img_loss;
                %% end of loading files
             
                    % generate optimal adv num vector 
                    two_stage_num = 3; % I set it to 4 to count for random ordering
                    % metrics = {'PGD-Step','Max-Gap','Min-Gap','Ave-Gap'};
                    metrics = {'PGD-Step'};
                    trans_rate_vec = adv_img_loss == 0;    
                    trans_rate = sum(trans_rate_vec)/length(trans_rate_vec);
                    trans_rate_cell(ii) = trans_rate;
                    all_adv_num_cell = cell(1,two_stage_num); % first is the strategy directly applies local info based order
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
                    target_greedy_cell{ii} = cat(1,greedy_init_adv_num',loss_greedy_adv_num_vec+greedy_init_adv_num(end)); 

                    % two stage strategy with only local info
                    trans_rate_vec_local_sorted = trans_rate_vec(local_sorted_order);
                    sorted_idx_local_info = local_sorted_order(~trans_rate_vec_local_sorted);
                    [local_greedy_adv_num_vec,~] = generate_adv_num_vec(query_num_vec,sorted_idx_local_info,succ_rate_vec,0,max_query_num);
                    % two stage strategy with local + target info
                    all_adv_num_cell{2} = cat(1,greedy_init_adv_num',local_greedy_adv_num_vec+greedy_init_adv_num(end)); 
                    local_greedy_cell{ii} = cat(1,greedy_init_adv_num',local_greedy_adv_num_vec+greedy_init_adv_num(end)); 

                    % add random ordering to all_adv_num_cell  
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
                    random_cell{ii} = cat(1,greedy_init_adv_num',rand_adv_num+greedy_init_adv_num(end)); 
                end
            end
            % take the mean of multiple runs
            target_greedy_tmp = 0;
            local_greedy_tmp = 0;
            random_tmp = 0;
            % collect quantitative info of opt strategy 
            additional_num_frac = [0.01,0.02,0.05,0.1];
            tar_greedy_quanti_info_mat = zeros(length(random_seed_vec),length(additional_num_frac)+1);
            local_greedy_quanti_info_mat = zeros(length(random_seed_vec),length(additional_num_frac)+1);
            random_quanti_info_mat = zeros(length(random_seed_vec),length(additional_num_frac)+1);
            for ii = 1:length(random_seed_vec)
                target_greedy_tmp = target_greedy_tmp + target_greedy_cell{ii};
                local_greedy_tmp = local_greedy_tmp + local_greedy_cell{ii};
                random_tmp = random_tmp + random_cell{ii};
                trans_rate = trans_rate_cell(ii);
                if (strcmp(attack_method,'nes') && strcmp(dataset,'cifar10') && strcmp(model,'madry_robust'))
                    fraction  = [trans_rate+0.01,trans_rate+0.02,trans_rate+0.03];
                elseif ~(strcmp(dataset,'mnist') && strcmp(model,'madry_robust'))
                    fraction  = [trans_rate+0.01,trans_rate+0.02,trans_rate+0.05,trans_rate+0.1];
                else
                    fraction  = [trans_rate,trans_rate+0.01];
                end
            
                for haha = 1:length(fraction)
                    tar_greedy_quanti_info_mat(ii,haha+1) = find(target_greedy_cell{ii} == int16(fraction(haha)*img_num),1);
                    local_greedy_quanti_info_mat(ii,haha+1)  = find(local_greedy_cell{ii} == int16(fraction(haha)*img_num),1);
                    random_quanti_info_mat(ii,haha+1)  = find(random_cell{ii} == int16(fraction(haha)*img_num),1);
                end
                tar_greedy_quanti_info_mat(ii,1) = find(target_greedy_cell{ii} == 1,1);
                local_greedy_quanti_info_mat(ii,1)  = find(local_greedy_cell{ii} == 1,1);
                random_quanti_info_mat(ii,1)  = find(random_cell{ii} == 1,1);
            end
            all_adv_num_cell{1} = round(target_greedy_tmp/length(random_seed_vec));
            all_adv_num_cell{2} = round(local_greedy_tmp/length(random_seed_vec));
            all_adv_num_cell{3} = round(random_tmp/length(random_seed_vec));
            
            % plot the graph containing optimal, random and different greedy strategies...
            legend_info = {'Prioritize on Target Info','Prioritize on Local Info','Random Order'};
            marker_info = {'-*','-s','-d'};
            fprintf('Now printing ALL scheduling strategies of Metric: %s\n',metric);
             
            % plot the greedy adv number plot
            figure;
            marker_index = round(query_budget/4);
            for ss = 1:length(all_adv_num_cell)-1
                adv_num = all_adv_num_cell{ss};
                plot(img_num:query_budget,adv_num(img_num+1:query_budget+1),marker_info{ss},'MarkerSize',12,'MarkerIndices', [marker_index],...
                'LineWidth',2);
                xlim([img_num query_budget]);
                ylim([0 Inf]);
                hold on;
                if query_budget <= 1000
                    xticks([0 100 200 300 400 500 600 700 800 900 1000])
                    xticklabels({'0' '100' '200' '300' '400' '500' '600' '700' '800' '900' '1000'})
                end
            end
            legend(legend_info,'FontSize', 18);
            xlabel('Budget on Number of Queries','FontSize', 18);
            ylabel('# of AEs Found','FontSize', 18);

            % print relevant info...
            fprintf('-------------Second Stage Quantitative Information: Metric: %s --------------------\n',metric);
            for ii = 1:length(fraction)+1
                if ii == 1
                    fprintf('First AE \n')
                    fprintf('Target Info: Mean (%2f) Std (%2f); Local Info: Mean (%2f) Std (%2f); Random: Mean (%2f) Std (%2f) \n',...
                        mean(tar_greedy_quanti_info_mat(:,1)),std(tar_greedy_quanti_info_mat(:,1)),...
                        mean(local_greedy_quanti_info_mat(:,1)),std(local_greedy_quanti_info_mat(:,1)),...
                        mean(random_quanti_info_mat(:,1)),std(random_quanti_info_mat(:,1)));
                else
                    fprintf('Additional %f of seeds:\n',additional_num_frac(ii-1))
                    fprintf('Target Info: Mean (%2f) Std (%2f); Local Info: Mean (%2f) Std (%2f); Random: Mean (%2f) Std (%2f) \n',...
                        mean(tar_greedy_quanti_info_mat(:,ii)),std(tar_greedy_quanti_info_mat(:,ii)),...
                        mean(local_greedy_quanti_info_mat(:,ii)),std(local_greedy_quanti_info_mat(:,ii)),...
                        mean(random_quanti_info_mat(:,ii)),std(random_quanti_info_mat(:,ii)));
                end
            end
            
          %% Begin of First Stage Processing 
            query_budget = img_num;
            first_stage_cell = cell(1,2);
            first_stage_cell{1} = all_adv_num_cell{2};
            tmp = all_adv_num_cell{2};
            first_stage_cell{2} = floor((0:img_num)*(max(tmp(1:query_budget+1))/(img_num))); 
            % first_stage_cell{3} = all_adv_num_cell{4};

            %plot the firts stages...
            figure;
            marker_index = round(query_budget/4);
            legend_info =  {'Transfer (PGD-Step)','Transfer (Random)'};
            for ss = 1:length(first_stage_cell)
                adv_num = first_stage_cell{ss};
                plot(0:query_budget,adv_num(1:query_budget+1),marker_info{ss+1},'MarkerSize',12,'MarkerIndices', [marker_index],...
                'LineWidth',2);
                xlim([1  query_budget]);
                ylim([0 Inf]);

                hold on;
                if query_budget <= 1000
                    xticks([0 100 200 300 400 500 600 700 800 900 1000])
                    xticklabels({'0' '100' '200' '300' '400' '500' '600' '700' '800' '900' '1000'})
                end
            end
            legend(legend_info,'FontSize', 18);
            xlabel('Budget on Number of Queries','FontSize', 18);
            ylabel('# of AEs Found','FontSize', 18);

            % print relevant info of first stage...  
            if ~(strcmp(dataset,'mnist') && strcmp(model,'madry_robust'))
                fraction  = [0.01,0.02,0.05];
            else
                fraction  = [0.01,0.02];
            end
            
            local_greedy_quanti_info_mat = zeros(length(random_seed_vec),length(fraction)+1);
            random_quanti_info_mat = zeros(length(random_seed_vec),length(fraction)+1);
            for ii = 1:length(random_seed_vec)
                trans_rate = trans_rate_cell(ii);
                rand_order_adv_num = floor((0:img_num)*trans_rate);
                for haha = 1:length(fraction)
                    local_greedy_quanti_info_mat(ii,haha+1)  = find(local_greedy_cell{ii} == fraction(haha)*img_num,1);
                    random_quanti_info_mat(ii,haha+1)  = find(rand_order_adv_num == fraction(haha)*img_num,1);
                end
                local_greedy_quanti_info_mat(ii,1)  = find(local_greedy_cell{ii} == 1,1);
                random_quanti_info_mat(ii,1)  = find(rand_order_adv_num == 1,1);
            end
            fprintf('---------------First Stage Quantitative Information: Metric: %s--------------------\n',metric);
            for ii = 1:length(fraction)+1
                if ii == 1
                    fprintf('First AE:\n')
                    fprintf('Local Info: Mean (%2f) Std (%2f); Random: Mean (%2f) Std (%2f) \n',...
                        mean(local_greedy_quanti_info_mat(:,1)),std(local_greedy_quanti_info_mat(:,1)),...
                        mean(random_quanti_info_mat(:,1)),std(random_quanti_info_mat(:,1)));
                else
                    fprintf('%f of seeds:\n',fraction(ii-1))
                    fprintf('Local Info: Mean (%2f) Std (%2f); Random: Mean (%2f) Std (%2f) \n',...
                        mean(local_greedy_quanti_info_mat(:,ii)),std(local_greedy_quanti_info_mat(:,ii)),...
                        mean(random_quanti_info_mat(:,ii)),std(random_quanti_info_mat(:,ii)));
                end
            end
          %% End of First Stage Processing
        end
    end
end
end