/**
* Triplet Sampling layer
* Strategy 1: each image as anchor, random select positive and negative.
*/

#include <algorithm>
#include <vector>
#include <map>
#include <iostream>
#include <stdlib.h>
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
	template <typename Dtype>
	void TripletSamplingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
	{
		CHECK_EQ(bottom[0]->width(), 1);
		CHECK_EQ(bottom[0]->height(), 1);
		
		CHECK_EQ(bottom[1]->channels(), 1);
		CHECK_EQ(bottom[1]->height(), 1);
		CHECK_EQ(bottom[1]->width(), 1);
		CHECK_EQ(bottom[1]->num(), bottom[0]->num());
		
		CHECK_EQ(top[0]->num(), top[1]->num());
		CHECK_EQ(top[1]->num(), top[2]->num());
		image_count.Reshape(bottom[0]->num(), 1, 1, 1);
	}
	template <typename Dtype>
	void TripletSamplingLayer<Dtype>::Reshape(
    	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), 1,1 );
		top[1]->Reshape(bottom[0]->num(), bottom[0]->channels(), 1,1 );
		top[2]->Reshape(bottom[0]->num(), bottom[0]->channels(), 1,1 );
		top0_map.Reshape(top[0]->num(), 1, 1, 1);
		top1_map.Reshape(top[1]->num(), 1, 1, 1);
		top2_map.Reshape(top[2]->num(), 1, 1, 1);
	}
	template <typename Dtype>
	void TripletSamplingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
	{
		//sampling strategy, generate top, record top0_map, top1_map, top2_map.
		const Dtype* bottom_label = bottom[1]->cpu_data();
		//store the label->dataIndex map.
		map<int, vector<int> > label_data_map;
		int max_label = 0;
		for(int i = 0; i < bottom[0]->num(); i++) {
			 const int label_value = static_cast<int>(bottom_label[i]);
			 if(label_value > max_label)
			 		max_label = label_value;
			 if(label_data_map.count(label_value) > 0){
			 	label_data_map[label_value].push_back(i);
			 }else{
			 	vector<int> tmp;
			 	tmp.push_back(i);
			 	label_data_map[label_value] = tmp;
			 }
		}
		if(label_data_map.size() == 1)
		{
			std::cout << "label number is 1" << std::endl;
			top[0]->Reshape(0, bottom[0]->channels(), 1,1 );
			top[1]->Reshape(0, bottom[0]->channels(), 1,1 );
			top[2]->Reshape(0, bottom[0]->channels(), 1,1 );
		}
		else{
			// for(map<int, vector<int> >::iterator it = label_data_map.begin(); it!=label_data_map.end(); ++it)
			// {
			// 	std::cout << "label:"  << it->first << ";size:" << it->second.size()<< std::endl;
			// }
			Dtype* anchors = top[0]->mutable_cpu_data();
			Dtype* positives = top[1]->mutable_cpu_data();
			Dtype* negatives = top[2]->mutable_cpu_data();
			int channels = bottom[0]->channels();
			Dtype* top0_map_Dtype = top0_map.mutable_cpu_data();
			Dtype* top1_map_Dtype = top1_map.mutable_cpu_data();
			Dtype* top2_map_Dtype = top2_map.mutable_cpu_data();
			for(int i = 0 ; i < bottom[0]->num(); i++)
			{
				const int label_value = static_cast<int>(bottom_label[i]);
				
				// caffe_cpu_axpby(
				// 				channels, 
				// 				Dtype(1), 
				// 				anchor,
				// 				Dtype(0.0),
				// 				anchors + (i * channels)
				// 				);
				//find positive data.
				int positive_index = i;
				if(label_data_map[label_value].size() != 1)
				{
					while(positive_index == i)
						positive_index = label_data_map[label_value][rand() % label_data_map[label_value].size()];
				}
				
				// caffe_cpu_axpby(
				// 				channels, 
				// 				Dtype(1), 
				// 				positive,
				// 				Dtype(0.0),
				// 				positives + (i * channels)
				// 				);
				//find negative data.
				int negative_label = label_value;
				while(negative_label == label_value || label_data_map.count(negative_label) == 0)
				{
					negative_label = rand() % (max_label + 1);
				}
				int negative_index = label_data_map[negative_label][rand() % label_data_map[negative_label].size()];
				// caffe_cpu_axpby(
				// 				channels, 
				// 				Dtype(1), 
				// 				negative,
				// 				Dtype(0.0),
				// 				negatives + (i * channels)
				// 				);
				const Dtype* anchor = bottom[0]->cpu_data() + (i * channels); 
				caffe_copy(channels, anchor, anchors + (i * channels));
				const Dtype* positive = bottom[0]->cpu_data() + (positive_index * channels);
				caffe_copy(channels, positive, positives + (i * channels));
				const Dtype* negative = bottom[0]->cpu_data() + (negative_index * channels);
				caffe_copy(channels, negative, negatives + (i * channels));

				top0_map_Dtype[i] = i;
				top1_map_Dtype[i] = positive_index;
				top2_map_Dtype[i] = negative_index;
			}
			Dtype* image_count_dtype = image_count.mutable_cpu_data();
			for(int i = 0 ; i < bottom[0]->num(); i++)
			{
				image_count_dtype[static_cast<int>(top0_map_Dtype[i])]++;
				image_count_dtype[static_cast<int>(top1_map_Dtype[i])]++;
				image_count_dtype[static_cast<int>(top2_map_Dtype[i])]++;
			}
		}
		
	}

	template <typename Dtype>
	void TripletSamplingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		int num = top[0]->num();
		int channels = top[0]->channels();
		//same image diff accumulate and average.
		if(propagate_down[0]){
			const Dtype* top0_map_Dtype = top0_map.cpu_data();
			const Dtype* top1_map_Dtype = top1_map.cpu_data();
			const Dtype* top2_map_Dtype = top2_map.cpu_data();
			Dtype* bout = bottom[0]->mutable_cpu_diff();
			const Dtype* image_count_dtype = image_count.cpu_data();
			for(int i = 0 ; i < num; ++i)
			{
				caffe_cpu_axpby(
							channels, 
					//		Dtype(1.0 / image_count_dtype[static_cast<int>(top0_map_Dtype[i])]), 
							Dtype(1.0),					
							top[0]->cpu_diff() + (i * channels),
							Dtype(0.0),
							bout + (static_cast<int>(top0_map_Dtype[i])*channels)
							);
				caffe_cpu_axpby(
							channels, 
							Dtype(1.0), 
							top[1]->cpu_diff() + (i * channels),
							Dtype(0.0),
							bout + (static_cast<int>(top1_map_Dtype[i])*channels)
							);	
				caffe_cpu_axpby(
							channels, 
							Dtype(1.0), 
							top[2]->cpu_diff() + (i * channels),
							Dtype(0.0),
							bout + (static_cast<int>(top2_map_Dtype[i])*channels)
							);	
			}
		}
	}
#ifdef CPU_ONLY
	STUB_GPU(TripletSamplingLayer);
#endif

INSTANTIATE_CLASS(TripletSamplingLayer);
REGISTER_LAYER_CLASS(TripletSampling);
}//namespace caffe
