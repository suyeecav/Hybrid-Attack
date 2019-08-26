function plot_graph_greedy_explore(query_budget,greedy_adv_num_cell,legend_info, marker_info,marker_index)
%plot the greedy_search_performance
figure;
max_adv_num = 0;
for i = 1:length(greedy_adv_num_cell) 
    adv_num = greedy_adv_num_cell{i};
    if max_adv_num < max(adv_num(1:query_budget+1))
        max_adv_num = max(adv_num(1:query_budget+1));
    end
end
    
for i = 1:length(greedy_adv_num_cell)
    adv_num = greedy_adv_num_cell{i};
    plot(0:query_budget,adv_num(1:query_budget+1),marker_info{i},'MarkerSize',12,'MarkerIndices', [marker_index],...
    'LineWidth',2);
    xlim([0  query_budget]);
    ylim([0 Inf]);
    hold on;
    %yticks([0 100 200 300 400 500 600 max_adv_num]);
    if query_budget <= 1000
        xticks([0 100 200 300 400 500 600 700 800 900 1000])
        xticklabels({'0' '100' '200' '300' '400' '500' '600' '700' '800' '900' '1000'})
    end
end
legend(legend_info,'FontSize', 18);
xlabel('Budget on Number of Queries','FontSize', 18);
ylabel('# of AEs Found','FontSize', 18);

% plot(0:query_budget,opt_adv_num(1:query_budget+1),0:query_budget,greedy_adv_num(1:query_budget+1),...
%     0:query_budget,rnd_adv_num(1:query_budget+1),...
%     'LineWidth',2);
% 
% 
% 
% %legend({'Greedy Search-Target Prob','Greedy Search-Loss Value','Random Search','Retroactive Optimal'},...
%  %   'FontSize', 12,'Location','northoutside','Orientation','horizontal');
% legend({'Retroactive Optimal Strategy','Greedy Approach (Hybrid)','Random (Hybrid)', 'FontSize', 12);
% xlabel('Budgets on Number of Queries','FontSize', 12);
% ylabel('Number of Adversarial Samples Found','FontSize', 12);
