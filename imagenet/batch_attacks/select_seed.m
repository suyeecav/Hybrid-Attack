function [sorted_order, candi_rate_vec] = select_seed(pgd_cnt_vec,img_loss,metric_type)
    max_val = 1e10;
    candi_rate_vec = pgd_cnt_vec < max_val;
    if strcmp(metric_type,'img-loss')
        [~,sorted_order] = sort(img_loss);
    elseif strcmp(metric_type,'pgd-step')
        sorted_order = [];
        sze = size(pgd_cnt_vec);
        not_selected_flag = true(sze(1),1);
        for i = 1:sze(2)
            curr_pgd_stp = pgd_cnt_vec(:,sze(2)-i+1);
            curr_stp_vec = curr_pgd_stp < max_val;
            if sum(curr_stp_vec) > 0
                update_flag =not_selected_flag & curr_stp_vec;
                idx = 1:sze(1);
                idx = idx(update_flag);
                q_num = curr_pgd_stp(update_flag);
                [~,sorted] = sort(q_num);
                sorted_order = cat(2,sorted_order,idx(sorted));
                not_selected_flag(update_flag) = 0;
            end
        end
        if length(sorted_order) < sze(1)
            idx = 1:sze(1);
            sorted_order = cat(2,sorted_order,idx(not_selected_flag));
        end
    elseif strcmp(metric_type,'mixture')
        % break ties when two pgd steps are identical to each other
        sorted_order = [];
        sze = size(pgd_cnt_vec);
        not_selected_flag = true(sze(1),1);
        for i = 1:sze(2)
            curr_pgd_stp = pgd_cnt_vec(:,sze(2)-i+1);
            
            curr_stp_vec = curr_pgd_stp < max_val;
            if i == sze(2)
                max_val_pgd_stp = curr_pgd_stp == max_val;
            end
            if sum(curr_stp_vec) > 0
                update_flag =not_selected_flag & curr_stp_vec;
                idx = 1:sze(1);
                idx = idx(update_flag);
                q_num = curr_pgd_stp(update_flag);
                
                % find all repetive elements
                for j = min(q_num) : max(q_num)
                    tmp_flag = q_num == j;
                    if sum(tmp_flag) > 1
                        % if ties occur
                        tie_idx = idx(tmp_flag);
                        [~,tie_idx_sort] = sort(img_loss(tie_idx));
                        sorted_order = cat(2,sorted_order,tie_idx(tie_idx_sort));
                    elseif sum(tmp_flag) == 1
                        sorted_order = cat(2,sorted_order,idx(tmp_flag));
                    end
                end
                if i == sze(2)
                    if sum(max_val_pgd_stp) > 0
                        update_flag = not_selected_flag & max_val_pgd_stp;
                        idx = 1:sze(1);
                        tie_idx = idx(update_flag);
                        [~,tie_idx_sort] = sort(img_loss(tie_idx));
                        sorted_order = cat(2,sorted_order,tie_idx(tie_idx_sort));
                    end
                end
                not_selected_flag(update_flag) = 0;
            end
        end
    end
end