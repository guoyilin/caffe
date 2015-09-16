#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void TripletSamplingLayer<Dtype>::Forward_gpu(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
//		std::cout << bottom[1]->count() << std::endl;
	//sampling strategy, generate top, record top0_map, top1_map, top2_map.
		Dtype* bottom_label = bottom[1]->mutable_cpu_data();
//		std::cout << "hello" << std::endl;
		//store the label->dataIndex map.
		map<int, vector<int> > label_data_map;
		int max_label = 0;
		for(int i = 0; i < bottom[0]->num(); i++) {
//			std::cout << "hello:" << static_cast<int>(bottom_label[i]) << std::endl;
			 const int label_value = static_cast<int>(bottom_label[i]);
			// std::cout << "hello2" << std::endl;
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
			Dtype* anchors = top[0]->mutable_cpu_data();
			Dtype* positives = top[1]->mutable_cpu_data();
			Dtype* negatives = top[2]->mutable_cpu_data();
			int channels = bottom[0]->channels();
			Dtype* top0_map_Dtype = top0_map.mutable_cpu_data();
			Dtype* top1_map_Dtype = top1_map.mutable_cpu_data();
			Dtype* top2_map_Dtype = top2_map.mutable_cpu_data();
			for(int i = 0 ; i < bottom[0]->num(); i++)
			{
//				std::cout << "hello" << std::endl;
				const int label_value = static_cast<int>(bottom_label[i]);
//				std::cout << "hello2" << std::endl;
				//find positive data.
				int positive_index = i;
				if(label_data_map[label_value].size() != 1)
				{
					while(positive_index == i)
						positive_index = label_data_map[label_value][rand() % label_data_map[label_value].size()];
				}
				//find negative data.
				int negative_label = label_value;
				while(negative_label == label_value || label_data_map.count(negative_label) == 0)
				{
					negative_label = rand() % (max_label + 1);
				}
				// std::cout << "hello2" << std::endl;
				int negative_index = label_data_map[negative_label][rand() % label_data_map[negative_label].size()];
				const Dtype* anchor = bottom[0]->cpu_data() + (i * channels); 
				caffe_copy(channels, anchor, anchors + (i * channels));
				const Dtype* positive = bottom[0]->cpu_data() + (positive_index * channels);
				caffe_copy(channels, positive, positives + (i * channels));
				const Dtype* negative = bottom[0]->cpu_data() + (negative_index * channels);
				caffe_copy(channels, negative, negatives + (i * channels));
				 //std::cout << "hello3" << std::endl;
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
/*
template <typenameDtype>
__global__ void CLLBackward(const int count, const int channels, const Dtype margin, const Dtype alpha, const Dtype* diff, const Dtype* dist_sq_ap_, const Dtype* dist_sq_an_, Dtype *bottom_diff)
{
	CUDA_KERNEL_LOOP(i, count) {
		int n = i/channels; 
		Dtype mdist(0.0);
		mdist = margin + dist_sq_ap_[n] - dist_sq_an_[n];
		if(mdist > 0.0){
			bottom_diff[i] = alpha * diff[i];
		}else{
			bottom_diff[i] = 0;
		}
	}
}
*/
template <typename Dtype>
void TripletSamplingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, 
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	int num = top[0]->num();
	int channels = top[0]->channels();
	//same image diff accumulate and average.
	if(propagate_down[0]){
		const Dtype* top0_map_Dtype = top0_map.mutable_cpu_data();
		const Dtype* top1_map_Dtype = top1_map.mutable_cpu_data();
		const Dtype* top2_map_Dtype = top2_map.mutable_cpu_data();
		Dtype* bout = bottom[0]->mutable_cpu_diff();
		const Dtype* image_count_dtype = image_count.mutable_cpu_data();
		for(int i = 0 ; i < num; ++i)
		{
			caffe_cpu_axpby(
						channels, 
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
INSTANTIATE_LAYER_GPU_FUNCS(TripletSamplingLayer);
}//namespace caffe
