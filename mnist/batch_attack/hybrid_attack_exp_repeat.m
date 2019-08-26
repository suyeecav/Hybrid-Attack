clear;clc;close all
%Note: this currently applies to MNIST and CIFAR10 datasets.
warning off;

set(0,'DefaultFigureVisible','on')

 %model_vec = {'densenet','madry_robust','mnist_robust','mnist',};
 model_vec = {'imagenet','mnist_robust','madry_robust',};
% model_vec = {'madry_robust','densenet'};
%target_type_vec = {'targeted','untargeted'};
target_type_vec = {'targeted'};
adv_type_vec = {'no_tune','norm'};
random_seed_vec = {'1234','2345','3456','4567','5678'};
loss_func = 'cw';
orig_class_vec = 0:9;
max_query_num = 4000;
class_num = 10;
img_num = 100;
average_name = 'repeat_5_times';
query_sum_collect = zeros(class_num,length(adv_type_vec));
prefix = '../..';
attack_method_vec = {'autozoom','nes'};
%attack_method_vec = {'autozoom'};
dist_metric = 'li';

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
        if strcmp(model,'mnist_robust')
             local_type_vec = {'simple_local'};
             rand_num = 100;
        elseif strcmp(model,'madry_robust') 
            local_type_vec = {'modelB_modelD_modelE','adv_densenet_adv_resnet'};
            rand_num = 100;
             %local_type_vec = {'modelB_modelD_modelE'};
         elseif strcmp(model,'imagenet') || strcmp(model,'imagenet_bandit')
             local_type_vec = {'repeated'};
             rand_num = 100;
             max_query_num = 100000;
        end
        different_local_query = cell(1,length(local_type_vec));
        different_local_succ = cell(1,length(local_type_vec));
        different_local_direct_transfer = cell(1,length(local_type_vec));
        
        succ_vec_cell = cell(1,length(local_type_vec));
        query_num_vec_cell = cell(1,length(local_type_vec));
        direct_trans_num_vec = cell(1,length(local_type_vec));

        for kk = 1:length(local_type_vec)
            fprintf('******************************************************\n')
            local_type = local_type_vec{kk};
            % store the performance in each independent run
            trans_rate_vec = zeros(1,length(random_seed_vec)); 
            orig_succ_rate = zeros(1,length(random_seed_vec)); 
            adv_succ_rate = zeros(1,length(random_seed_vec)); 
            orig_query_ae = zeros(1,length(random_seed_vec)); 
            adv_query_ae = zeros(1,length(random_seed_vec)); 
            orig_query_seed = zeros(1,length(random_seed_vec)); 
            adv_query_seed = zeros(1,length(random_seed_vec)); 
            orig_query_search = zeros(1,length(random_seed_vec)); 
            adv_query_search = zeros(1,length(random_seed_vec)); 
            mean_cost_reduction_vec = zeros(1,length(random_seed_vec)); 
            violation_vec = zeros(1,length(random_seed_vec)); 
            
            for ii = 1:length(random_seed_vec)
                rand_seed = random_seed_vec{ii};
                violation_check = cell(1,2);
                succ_vec_check = cell(1,2);
                % find direct transfers with adv image loss vectors    
                if strcmp(model,'mnist') || strcmp(model,'mnist_robust')
                    file_path_head = [prefix '/sub_saved_qual/' model '/' target_type '/'  local_type '/' loss_func '/' average_name '/' rand_seed '/'];
                else
                    file_path_head = [prefix '/sub_saved_qual/' model '/' target_type '/' local_type '/' loss_func '/' average_name '/' rand_seed '/'];
                end
                if strcmp(model,'imagenet')
                    adv_loss_name = ['adv_img_loss_100.txt'];
                else
                    adv_loss_name = ['adv_img_loss_' num2str(rand_num) '.txt'];
                end
                
                file_full_path=fullfile(file_path_head,adv_loss_name);
                fileID = fopen(file_full_path);
                formatSpec = '%f';
                adv_img_loss = fscanf(fileID,formatSpec);
                direct_transfer_idx = adv_img_loss == 0;

                for k = 1:length(adv_type_vec)
                    adv_type = adv_type_vec{k};
                    % load query number and success rate vec
                    fprintf('-------------Printing %s %s %s %s %s Round %d:-----------------\n', attack_method,target_type,model,local_type,adv_type,ii);
                    file_path_head1 = [prefix '/' attack_method '/' model '/' target_type '/' local_type '/' loss_func '/' average_name '/' rand_seed '/'];
                    if strcmp(model,'densenet') || strcmp(model,'madry_robust')
                        c_val_str = '_cval0.05_';
                    elseif strcmp(model,'mnist') || strcmp(model,'mnist_robust')
                        c_val_str = '_cval0.3_';
                    elseif strcmp(model,'imagenet')
                        c_val_str = '_cval12.0_';
                    end
                    query_num_name = ['query_num_vec_all' c_val_str adv_type '_random_' num2str(rand_num) '.txt'];
                    %query_num_name = ['query_num_vec_all' c_val_str adv_type '_random.txt'];
                    file_full_path=fullfile(file_path_head1,query_num_name);
                    fileID = fopen(file_full_path);
                    formatSpec = '%f';
                    query_num_vec = fscanf(fileID,formatSpec);
                    if strcmp(adv_type,'no_tune')
                        query_num_vec(direct_transfer_idx) = 1;
                    end

                    if min(query_num_vec) == 0
                        query_num_vec = query_num_vec + 1;
                    elseif strcmp(attack_method,'autozoom') && strcmp(adv_type,'norm')
                        query_num_vec = query_num_vec + 1;
                    end
                    if max(query_num_vec) > max_query_num
                        query_num_vec(query_num_vec>max_query_num) = max_query_num; 
                    end

                    % success rate vec
                    succ_rate_name = ['succ_vec_all' c_val_str adv_type '_random_' num2str(rand_num) '.txt'];
                    %succ_rate_name = ['succ_vec_all' c_val_str adv_type '_random.txt'];
                    file_full_path=fullfile(file_path_head1,succ_rate_name);
                    fileID = fopen(file_full_path);
                    formatSpec = '%f';
                    succ_rate_vec = fscanf(fileID,formatSpec);
                    succ_query_num_vec = query_num_vec(logical(succ_rate_vec));
                    % info of non-directly transferrable examples
                    non_trans_query_num = query_num_vec(~direct_transfer_idx);
                    non_trans_succ_rate_vec = succ_rate_vec(~direct_transfer_idx);
                    trans_rate = sum(direct_transfer_idx)/length(direct_transfer_idx);
                    if strcmp(adv_type,'norm')
                        violation_check{1} = query_num_vec;
                        succ_vec_check{1} = succ_rate_vec;
                    elseif strcmp(adv_type,'no_tune')
                        violation_check{2} = query_num_vec;
                        succ_vec_check{2} = succ_rate_vec;
                    end
                    
                    if strcmp(adv_type,'no_tune')
                        succ_vec_cell{kk} = succ_rate_vec;
                        query_num_vec_cell{kk} = query_num_vec;
                        direct_trans_num_vec{kk} = direct_transfer_idx;
                    end
                    
                    query_ae = sum(query_num_vec)/sum(succ_rate_vec);
                    query_seed = sum(query_num_vec)/length(query_num_vec);
                    query_search = sum(non_trans_query_num)/sum(non_trans_succ_rate_vec);
                    succ_rate = sum(succ_rate_vec)/length(succ_rate_vec);
                    
                    % record query info of different locals
                    if strcmp(adv_type,'no_tune')
                        different_local_query{kk} = query_num_vec;
                        different_local_succ{kk} = succ_rate_vec; 
                        different_local_direct_transfer{kk} = direct_transfer_idx;
                    end
        
                    fprintf('Query/AE: %.3f; Query/Seed: %.3f, Success Rate %.3f; Transfer Rate %f Non-transfer Mean Cost: %.3f Non-Transfer Succ Rate: %.3f \n',...
                       query_ae, query_seed,succ_rate,trans_rate,query_search,sum(non_trans_succ_rate_vec)/length(non_trans_succ_rate_vec));

                   % store the cost info of individual runs
                    if strcmp(adv_type,'no_tune')
                        trans_rate_vec(ii) = trans_rate;
                    end
                    if strcmp(adv_type,'norm')
                        orig_succ_rate(ii) = succ_rate;
                        orig_query_seed(ii) = query_seed;
                        orig_query_ae(ii) = query_ae;
                        orig_query_search(ii) = query_search;
                    elseif strcmp(adv_type,'no_tune')
                        adv_succ_rate(ii) = succ_rate;
                        adv_query_seed(ii) = query_seed;
                        adv_query_ae(ii) = query_ae;
                        adv_query_search(ii) = query_search;
                    end
                    
                    % demonstrate the query number variance
                    if strcmp(model,'mnist') && strcmp(adv_type,'no_tune') && strcmp(local_type,'simple_local')
                        mnist_cifar10_query{1} = succ_query_num_vec;
                    elseif strcmp(model,'densenet') && strcmp(adv_type,'no_tune') && strcmp(local_type,'modelB_modelD_modelE')
                        mnist_cifar10_query{2} = succ_query_num_vec;
                    end
                end
                violation_val = violation_check{1} < violation_check{2};
                orig_ae_cost = sum(violation_check{1})/sum(succ_vec_check{1});
                adv_ae_cost = sum(violation_check{2})/sum(succ_vec_check{2});
                mean_cost_reduction = sum(orig_ae_cost-adv_ae_cost)/orig_ae_cost;
                fprintf('violation percentage %f; mean cost reduction: %f \n',sum(violation_val)/length(violation_val),mean_cost_reduction);
                mean_cost_reduction_vec(ii) = mean_cost_reduction;
                violation_vec(ii) = sum(violation_val)/length(violation_val);
            end
            % now print the info of 5 independent runs
            fprintf('Direct transfer rate: Mean (%3f) Std (%3f) \n',mean(trans_rate_vec),std(trans_rate_vec));
            fprintf(' ## Adv ##  Success Rate: Mean (%.3f) Std (%.3f); Query/AE: Mean (%.3f) Std (%.3f); Query/Seed: Mean (%.3f) Std (%.3f); Query/Search: Mean (%.3f) Std (%.3f) \n',...
            mean(adv_succ_rate),std(adv_succ_rate),mean(adv_query_ae),std(adv_query_ae),mean(adv_query_seed),std(adv_query_seed),mean(adv_query_search),std(adv_query_search));            
            fprintf(' ## Orig ##  Success Rate: Mean (%.3f) Std (%.3f); Query/AE: Mean (%.3f) Std (%.3f); Query/Seed: Mean (%.3f) Std (%.3f); Query/Search: Mean (%.3f) Std (%.3f) \n',...
            mean(orig_succ_rate),std(orig_succ_rate),mean(orig_query_ae),std(orig_query_ae),mean(orig_query_seed),std(orig_query_seed),mean(orig_query_search),std(orig_query_search));
            fprintf('Mean Cost Reduction:  Mean (%.3f) Std (%.3f), Violation: Mean (%.3f) Std (%.3f) \n',mean(mean_cost_reduction_vec),...
                std(mean_cost_reduction_vec),mean(violation_vec), std(violation_vec));
        end
        
     % compare different local models
         max_num = 0;
         for kk = 1:length(direct_trans_num_vec)
             if max_num < sum(direct_trans_num_vec{kk})
                 max_num = sum(direct_trans_num_vec{kk});
                 tmp_direct_transfer = direct_trans_num_vec{kk};
             end
         end
         for kk = 1:length(direct_trans_num_vec)
             tmp_query_num_vec = different_local_query{kk};
             tmp_query_num_vec = tmp_query_num_vec(~tmp_direct_transfer);
             tmp_succ_vec = different_local_succ{kk};
             tmp_succ_vec = tmp_succ_vec(~tmp_direct_transfer);
             fprintf('(strict non-transfers) mean cost of local model %s is %f \n',local_type_vec{kk},sum(tmp_query_num_vec)/sum(tmp_succ_vec));
         end
         min_num = 1e20;
         for kk = 1:length(direct_trans_num_vec)
             if min_num > sum(direct_trans_num_vec{kk})
                 min_num = sum(direct_trans_num_vec{kk});
                 tmp_direct_transfer = direct_trans_num_vec{kk};
             end
         end
         for kk = 1:length(direct_trans_num_vec)
             tmp_query_num_vec = different_local_query{kk};
             tmp_query_num_vec = tmp_query_num_vec(~tmp_direct_transfer);
             tmp_succ_vec = different_local_succ{kk};
             tmp_succ_vec = tmp_succ_vec(~tmp_direct_transfer);
             fprintf('(loose non-transfer) mean cost of local model %s is %f \n',local_type_vec{kk},sum(tmp_query_num_vec)/sum(tmp_succ_vec));
         end    

end
% plot the query distribution
figure;
%legend_info = {'MNIST','CIFAR10'};
marker = {'-*','-s'};
for haha = 1:length(mnist_cifar10_query)
    plot(1:length(mnist_cifar10_query{haha}),sort(mnist_cifar10_query{haha}), marker{haha},'MarkerIndices', [700 800],...
        'MarkerSize',12, 'LineWidth',2);
    ylim([1 4000]);
    xlim([1 1000]);
    xticks([1 200 400 600 800 1000]);
    yticks([1 500 1000 1500 2000 2500 3000 3500 4000]);

    xlabel('Images Sorted by Number of Queries','FontSize', 16);
    ylabel('Number of Queries','FontSize', 16);
    hold on;
end
    %legend(legend_info, 'FontSize', 18);
    hold off;
end
