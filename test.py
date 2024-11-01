class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        num_count_tab = dict()
        count_elements_tab = dict()
        k_tab = dict()
        index_k = 0
        for i in nums:
            if i in num_count_tab:
                num_count_tab[i] +=1
            else:
                num_count_tab[i] =1
            # if num_count_tab[i] > max_count:
            #     max_count = num_count_tab[i]
            
            if num_count_tab[i] in count_elements_tab:
                count_elements_tab[num_count_tab[i]].append(i)
                k_tab[num_count_tab[i]] += 1
                if k_tab[num_count_tab[i]] == k:
                    results = count_elements_tab[num_count_tab[i]]
            else:
                count_elements_tab[num_count_tab[i]] = [i]
                k_tab[num_count_tab[i]] = 1
                if k_tab[num_count_tab[i]] == k:
                    results = count_elements_tab[num_count_tab[i]]
            
        # flipped_dict = dict(zip(num_count_tab.values(), num_count_tab.keys()))
        # sorted_k = sorted(flipped_dict.keys(),reverse=True)
        # results = []
        # for i in range(0,k):
        #     results.append(flipped_dict[sorted_k[i]])
        return results
    
nums = [1,1,1,2,2,3]
k = 2
sol = Solution()
print(sol.topKFrequent(nums,k))    
