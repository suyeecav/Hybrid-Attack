function [greedy_adv_num_vec,...
    query_budget_vec] = generate_adv_num_vec(query_num_vec,ins_order,succ_vec,cost_base,max_query_num)
% :param: cost_base: to count for the initial query spent on obtaining the
% initial predictions scores (and the loss function)

sorted_query =  query_num_vec(ins_order);
sorted_succ_vec = succ_vec(ins_order);
inst_num = length(sorted_query);
inst_num = 1000;
%query budget vector generation...
total_query_num = inst_num*max_query_num;
query_budget_vec = 0:total_query_num;

%Now generate the number of sucessful instances w.r.t query budget
greedy_adv_num_vec = zeros(length(query_budget_vec),1);
% cost_base = 1;
adv_count = 0;

for ii = 1:length(sorted_query)
  cost = sorted_query(ii);  
  if (cost_base + cost) <= total_query_num+1
      greedy_adv_num_vec(cost_base+1:cost_base+cost - 1) = adv_count;
      if logical(sorted_succ_vec(ii))
          adv_count = adv_count + 1;
      end
      greedy_adv_num_vec(cost_base+cost) = adv_count;
  else
      greedy_adv_num_vec(cost_base+1:total_query_num+1) = adv_count;
      break;
  end
  cost_base = cost_base + cost;%record number of total queries spent
end

if greedy_adv_num_vec(end) == 0%indicates that total number of queries are available more
    greedy_adv_num_vec(cost_base+1:total_query_num+1) = adv_count;
end