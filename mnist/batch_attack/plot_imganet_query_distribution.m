close all;clc;clear
prefix = '..';
attack_method = 'autozoom';
model = 'imagenet';
target_type = 'targeted';
prefix1 = 'repeated';
file_path_head1 = [prefix '/' attack_method '/' model '/' target_type '/' prefix1 '/'];
 c_val_str = '_12.0_';
adv_type = 'no_tune';
img_indices = {'0_100', '100_200', '200_300', '300_400', '400_500'};
max_query_limit = 100000;
query_num_vec_all = [];
orig_query_num_vec_all = [];
for i =1:length(img_indices)
    % query_num_name = ['query_num_vec_all' c_val_str adv_type '.txt'];
    img_idx = img_indices{i};
    query_num_name = ['adv/adv_num_queries' c_val_str img_idx '.txt'];
    file_full_path=fullfile(file_path_head1,query_num_name);
    fileID = fopen(file_full_path);
    formatSpec = '%f';
    query_num_vec = fscanf(fileID,formatSpec);
    query_num_vec_all = cat(1,query_num_vec_all,query_num_vec);
    
    % load the adv query number
    query_num_name = ['orig/orig_num_queries' c_val_str img_idx '.txt'];
    file_full_path=fullfile(file_path_head1,query_num_name);
    fileID = fopen(file_full_path);
    formatSpec = '%f';
    orig_query_num_vec = fscanf(fileID,formatSpec);
    orig_query_num_vec_all = cat(1,orig_query_num_vec_all,orig_query_num_vec);
end

% adv_img_loss
adv_img_loss_name = ['adv/adv_img_loss_12.0.txt'];
file_full_path=fullfile(file_path_head1,adv_img_loss_name);
fileID = fopen(file_full_path);
formatSpec = '%f';
adv_img_loss = fscanf(fileID,formatSpec);
direct_transfers = adv_img_loss == 0;

query_num_vec_all = sort(query_num_vec_all);
non_trans_query_num_vec_all = query_num_vec_all((~direct_transfers)&(query_num_vec_all<max_query_limit));

max_val = max(non_trans_query_num_vec_all);

figure;
%semilogy(1:length(non_trans_query_num_vec_all),sort(query_num_vec+1),'LineWidth',2);
plot(1:length(non_trans_query_num_vec_all),non_trans_query_num_vec_all,'m','LineWidth',2);
set(gca,'FontSize',20);
ylim([1 max(non_trans_query_num_vec_all)]);
xlim([1 length(non_trans_query_num_vec_all)]);
xticks([1 100 200 300 400 length(non_trans_query_num_vec_all)]);
%ytickformat('%.5d') 
%yticks([1 10000 20000 30000 40000 50000 60000 70000 80000 90000 max_val])
yticks([1 20000 40000 60000 80000 max_val])
%yticklabels({'1', '10000', '20000', '30000', '40000', '50000', '60000', '70000', '80000', '90000', num2str(max_val)});
yticklabels({'1', '20000', '40000', '60000', '80000', num2str(max_val)});

%legend({'ImageNet'}, 'FontSize', 18);
%xlabel('Images Sorted by Number of Queries','FontSize', 16);
%ylabel('Number of Queries','FontSize', 16);
top_n_fraction  = 0.1;
top_n = top_n_fraction * length(non_trans_query_num_vec_all);
mean(non_trans_query_num_vec_all(1:top_n))
mean(non_trans_query_num_vec_all)

% check the variance of each independent run:
iid_run = 5;
img_num = 100;
for i =1:iid_run
    iid_query_num = query_num_vec_all((i-1)*img_num+1:i*img_num);
    succ_rate = iid_query_num < max_query_limit;
    ave_query = sum(iid_query_num) / sum(succ_rate);
    
    iid_orig_query_num = orig_query_num_vec_all((i-1)*img_num+1:i*img_num);
    orig_succ_rate = iid_orig_query_num < max_query_limit;
    orig_ave_query = sum(iid_orig_query_num)/sum(orig_succ_rate);
    fprintf('succ rate of adv is %f and orig is %f\n',sum(succ_rate)/length(succ_rate),sum(orig_succ_rate/length(orig_succ_rate)))
    fprintf('ave cost of adv is %f, orig is %f \n',ave_query,orig_ave_query);
end
