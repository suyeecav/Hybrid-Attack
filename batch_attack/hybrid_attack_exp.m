clear;clc;close all
%Note: this currently applies to MNIST and CIFAR10 datasets.
warning off;

set(0,'DefaultFigureVisible','on')

 %model_vec = {'densenet','madry_robust','mnist_robust','mnist',};
 model_vec = {'densenet','madry_robust','mnist_robust'};
% model_vec = {'madry_robust','densenet'};
%target_type_vec = {'targeted','untargeted'};
target_type_vec = {'targeted'};
adv_type_vec = {'no_tune','norm'};
loss_func = 'cw';
orig_class_vec = 0:9;
class_num = 10;
img_num = 100;
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
        if strcmp(model,'mnist') || strcmp(model,'mnist_robust')
             local_type_vec = {'simple_local'};
        elseif strcmp(model,'madry_robust') || strcmp(model,'densenet')
            local_type_vec = {'modelB_modelD_modelE','adv_densenet_adv_resnet'};
            %local_type_vec = {'modelB_modelD_modelE'};
        end
        
%         if strcmp(model,'mnist')
%              tar_class_vec = [2,9,3,5,5,3,5,9,3,4];
%              local_type_vec = {'simple_local'};
%         elseif strcmp(model,'mnist_robust')
%             tar_class_vec =orig_class_vec;
%             local_type_vec = {'simple_local'};
%         elseif strcmp(model,'cifar10') 
%             tar_class_vec = [6,2,8,0,0,0,7,8,4,4];
%              %local_type_vec = {'simple_local','complex_local'};
%              local_type_vec = {'simple_local'};
%         elseif strcmp(model,'cifar10_robust')
%             tar_class_vec = orig_class_vec;
%              %local_type_vec = {'simple_local','complex_local'};
%              local_type_vec = {'simple_local'};
%         end

        succ_vec_cell = cell(1,length(local_type_vec));
        query_num_vec_cell = cell(1,length(local_type_vec));
        direct_trans_num_vec = cell(1,length(local_type_vec));
        for kk = 1:length(local_type_vec)
            fprintf('******************************************************\n')
            local_type = local_type_vec{kk};
            violation_check = cell(1,2);
            % find direct transfers with adv image loss vectors    
            if strcmp(model,'mnist') || strcmp(model,'mnist_robust')
                file_path_head = [prefix '/sub_saved_qual/' model '/' target_type '/' local_type '/' loss_func '/'];
            else
                file_path_head = [prefix '/sub_saved_qual/' model '/' target_type '/' local_type '/' loss_func '/'];
            end
            adv_loss_name = ['adv_img_loss.txt'];
            file_full_path=fullfile(file_path_head,adv_loss_name);
            fileID = fopen(file_full_path);
            formatSpec = '%f';
            adv_img_loss = fscanf(fileID,formatSpec);
            direct_transfer_idx = adv_img_loss == 0;
            
            for k = 1:length(adv_type_vec)
                adv_type = adv_type_vec{k};
                % load query number and success rate vec
                if (strcmp(model,'densenet') || strcmp(model,'madry_robust'))&&strcmp(adv_type,'norm')
                    %local_type = 'no_local_model';
%                 elseif strcmp(model,'mnist') || strcmp(model,'mnist_robust')
%                     %file_path_head1 = [prefix '/' attack_method '/' model '/' target_type '/'];
%                     local_type = 'no_local_model';
                end
                fprintf('--------------Printing %s %s %s %s %s:-----------------\n', attack_method,target_type,model,local_type,adv_type);
                file_path_head1 = [prefix '/' attack_method '/' model '/' target_type '/' local_type '/' loss_func '/'];
                if strcmp(model,'densenet') || strcmp(model,'madry_robust')
                    c_val_str = '_cval0.05_';
                elseif strcmp(model,'mnist') || strcmp(model,'mnist_robust')
                    c_val_str = '_cval0.3_';
                end
                %query_num_name = ['query_num_vec_all' c_val_str adv_type '_random.txt'];
                if strcmp(model,'madry_robust') || strcmp(model,'mnist_robust')
                    query_num_name = ['momentum/query_num_vec_all' c_val_str adv_type '_random.txt'];
                else
                    query_num_name = ['query_num_vec_all' c_val_str adv_type '_random.txt'];
                end
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
                
                direct_transfer = sum(query_num_vec == 1)/length(query_num_vec);
                if strcmp(adv_type,'norm')
                    violation_check{1} = query_num_vec;
                elseif strcmp(adv_type,'no_tune')
                    violation_check{2} = query_num_vec;
                end
                % success rate vec
                %succ_rate_name = ['succ_vec_all' c_val_str adv_type '_random.txt'];
                if strcmp(model,'madry_robust') || strcmp(model,'mnist_robust')
                    succ_rate_name = ['momentum/succ_vec_all' c_val_str adv_type '_random.txt'];
                else
                    succ_rate_name = ['succ_vec_all' c_val_str adv_type '_random.txt'];
                end
                file_full_path=fullfile(file_path_head1,succ_rate_name);
                fileID = fopen(file_full_path);
                formatSpec = '%f';
                succ_rate_vec = fscanf(fileID,formatSpec);
                succ_query_num_vec = query_num_vec(logical(succ_rate_vec));
                % info of non-directly transferrable examples
                non_trans_query_num = query_num_vec(~direct_transfer_idx);
                non_trans_succ_rate_vec = succ_rate_vec(~direct_transfer_idx);
                if strcmp(adv_type,'no_tune')
                    succ_vec_cell{kk} = succ_rate_vec;
                    query_num_vec_cell{kk} = query_num_vec;
                    direct_trans_num_vec{kk} = direct_transfer_idx;
                end
                
                fprintf('All mean: %.2f AE Mean: %.2f; success rate %f; direct transfer %f Non-transfer mean cost: %3f Non-Transfer succ rate: %3f \n',...
                   sum(query_num_vec)/length(query_num_vec),sum(query_num_vec)/sum(succ_rate_vec), sum(succ_rate_vec)/length(succ_rate_vec),direct_transfer,...
                   sum(non_trans_query_num)/sum(non_trans_succ_rate_vec),sum(non_trans_succ_rate_vec)/length(non_trans_succ_rate_vec));
                
                % demonstrate the query number variance
                if strcmp(model,'mnist') && strcmp(adv_type,'no_tune') && strcmp(local_type,'simple_local')
                    mnist_cifar10_query{1} = succ_query_num_vec;
                elseif strcmp(model,'densenet') && strcmp(adv_type,'no_tune') && strcmp(local_type,'modelB_modelD_modelE')
                    mnist_cifar10_query{2} = succ_query_num_vec;
                end
                
            end
            violation_vec = violation_check{1} < violation_check{2};
            fprintf('violation percentage %f \n',sum(violation_vec)/length(violation_vec));
        end
        max_num = 0;
        for kk = 1:length(direct_trans_num_vec)
            if max_num < sum(direct_trans_num_vec{kk})
                max_num = sum(direct_trans_num_vec{kk});
                tmp_direct_transfer = direct_trans_num_vec{kk};
            end
        end
        for kk = 1:length(direct_trans_num_vec)
            tmp_query_num_vec = query_num_vec_cell{kk};
            tmp_query_num_vec = tmp_query_num_vec(~tmp_direct_transfer);
            tmp_succ_vec = succ_vec_cell{kk};
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
            tmp_query_num_vec = query_num_vec_cell{kk};
            tmp_query_num_vec = tmp_query_num_vec(~tmp_direct_transfer);
            tmp_succ_vec = succ_vec_cell{kk};
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
