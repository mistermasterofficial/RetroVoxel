//[f8[:], f8, f8, i8, i8, i8[:,:,:], i8[:,:,:,:], i8]
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>

extern "C"
{
	//class Screens_cpp
	//{
	//public:
	//	int* screens;
	//	int width;
	//	int height;
	//	int nums;

	//	Screens_cpp(int _width, int _height, int _nums) : width(_width), height(_height), nums(_nums)
	//	{
	//		screens = (int*)malloc(_width * _height * _nums * 3 * sizeof(int));
	//	}
	//	~Screens_cpp()
	//	{
	//		free(screens);
	//	}

	//	int get_pixel(int x, int y, int n, int rgb)
	//	{
	//		return screens[n * width * height + x * height + y + rgb];
	//	}

	//	void set_pixel(int x, int y, int n, int rgb, int val)
	//	{
	//		screens[n * width * height + x * height + y + rgb] = val;
	//	}
	//};

	//class Scene_cpp
	//{
	//public:
	//	int* field;
	//	int field_size_z;
	//	int field_size_y;
	//	int field_size_x;
	//	int* assets;
	//	int assets_num_blocks;
	//	int resolution;

	//	Scene_cpp(
	//		int _field_size_z,
	//		int _field_size_y,
	//		int _field_size_x,

	//		int _resolution,
	//		int _assets_num_blocks
	//		) : field_size_z(_field_size_z), field_size_y(field_size_y), field_size_x(field_size_x), resolution(_resolution), assets_num_blocks(_assets_num_blocks)
	//	{
	//		field = (int*)malloc(_field_size_z * _field_size_y * _field_size_x * sizeof(int));
	//		assets = (int*)malloc(_assets_num_blocks * _resolution * _resolution * _resolution * 3 * sizeof(int));
	//	}

	//	~Scene_cpp()
	//	{
	//		free(field);
	//		free(assets);
	//	}

	//	void set_field(int x, int y, int z, int val)
	//	{
	//		field[z * field_size_x * field_size_y + y * field_size_x + x] = val;
	//	}

	//	void set_asset(int num_block, int z, int x, int y, int rgb, int val)
	//	{
	//		//assets[num_block][z][x][y][rgb] = val;
	//		assets[num_block * resolution * resolution * resolution + z * resolution * resolution + x * resolution + y + rgb] = val;
	//	}

	//	int get_field(int x, int y, int z)
	//	{
	//		return field[z*field_size_x*field_size_y+y*field_size_x+x];
	//	}

	//	int get_asset(int num_block, int z, int x, int y, int rgb)
	//	{
	//		return assets[num_block*resolution*resolution*resolution+z*resolution*resolution+x*resolution+y+rgb];
	//	}

	//	Screens_cpp render(float camera_pos_x, float camera_pos_y, float angle, float fov, int ray_num, int distance)
	//	{
	//		Screens_cpp result(ray_num, resolution * field_size_z, distance * resolution);

	//		for (int v = 0; v < ray_num; v++)
	//		{
	//			float static_vec_x = (float)(cos((double)((angle - fov / 2.0 + fov / (ray_num - 1) * (float)v) / 180.0 * M_PI)) / cos((double)((-fov / 2.0 + fov / (ray_num - 1) * (float)v) / 180.0 * M_PI)));
	//			float static_vec_y = (float)(sin((double)((angle - fov / 2.0 + fov / (ray_num - 1) * (float)v) / 180.0 * M_PI)) / cos((double)((-fov / 2.0 + fov / (ray_num - 1) * (float)v) / 180.0 * M_PI)));
	//			for(int z = 0; z < distance * resolution; z++)
	//			{
	//				float vec_x = static_vec_x * (float)(z+1) / (float)(resolution)+camera_pos_x;
	//				float vec_y = static_vec_y * (float)(z + 1) / (float)(resolution)+camera_pos_y;
	//				if ((int)(vec_x) >= 0 && (int)(vec_y) && (int)(vec_x) < field_size_x && (int)(vec_y) < field_size_y)
	//				{
	//					for (int i = 0; i < resolution * field_size_z; i++)
	//					{
	//						for (int rgb = 0; rgb < 3; rgb++)
	//						{
	//							result.set_pixel(v, i, z, rgb, get_asset(get_field((int)vec_x,(int)vec_y,i/resolution),i%resolution,(int)vec_y*resolution%resolution, (int)vec_x * resolution % resolution, rgb));
	//						}
	//					}
	//				}
	//			}
	//		}

	//		return result;
	//	}
	//};

struct Screens_cpp
{
	int* screens;
	int width;
	int height;
	int nums;
};

Screens_cpp* get_screens(int _width, int _height, int _nums)
{
	Screens_cpp* res = new Screens_cpp;
	res->width = _width;
	res->height = _height;
	res->nums = _nums;
	res->screens = (int*)malloc(_width * _height * _nums * 3 * sizeof(int));

	return res;
}

void del_screens(Screens_cpp* scr)
{
	free(scr->screens);
}

int get_pixel(Screens_cpp* scr, int x, int y, int n, int rgb)
{
	return scr->screens[n * scr->width * scr->height + x * scr->height + y + rgb];
}

void set_pixel(Screens_cpp* scr, int x, int y, int n, int rgb, int val)
{
	scr->screens[n * scr->width * scr->height + x * scr->height + y + rgb] = val;
}

struct Scene_cpp
{
	int* field;
	int field_size_z;
	int field_size_y;
	int field_size_x;
	int* assets;
	int assets_num_blocks;
	int resolution;
};

Scene_cpp* get_scene(int _field_size_z,int _field_size_y,int _field_size_x,int _resolution,int _assets_num_blocks)
{
	Scene_cpp* res = new Scene_cpp;
	res->field_size_x = _field_size_x;
	res->field_size_y = _field_size_y;
	res->field_size_z = _field_size_z;
	res->resolution = _resolution;
	res->assets_num_blocks = _assets_num_blocks;

	res->field = (int*)malloc(_field_size_z * _field_size_y * _field_size_x * sizeof(int));
	res->assets = (int*)malloc(_assets_num_blocks * _resolution * _resolution * _resolution * 3 * sizeof(int));

	return res;
}

void del_scene(Scene_cpp* scene)
{
	free(scene->field);
	free(scene->assets);
}

void set_field(Scene_cpp* scene, int x, int y, int z, int val)
{
	scene->field[z * scene->field_size_x * scene->field_size_y + y * scene->field_size_x + x] = val;
}

void set_asset(Scene_cpp* scene, int num_block, int z, int x, int y, int rgb, int val)
{
	scene->assets[num_block * scene->resolution * scene->resolution * scene->resolution + z * scene->resolution * scene->resolution + x * scene->resolution + y + rgb] = val;
}

int get_field(Scene_cpp* scene, int x, int y, int z)
{
	return scene->field[z * scene->field_size_x * scene->field_size_y + y * scene->field_size_x + x];
}

int get_asset(Scene_cpp* scene, int num_block, int z, int x, int y, int rgb)
{
	return scene->assets[num_block * scene->resolution * scene->resolution * scene->resolution + z * scene->resolution * scene->resolution + x * scene->resolution + y + rgb];
}

Screens_cpp* render(Scene_cpp* scene, float camera_pos_x, float camera_pos_y, float angle, float fov, int ray_num, int distance)
{
	Screens_cpp* result = get_screens(ray_num, scene->resolution * scene->field_size_z, distance * scene->resolution);

	for (int v = 0; v < ray_num; v++)
	{
		float static_vec_x = (float)(cos((double)((angle - fov / 2.0 + fov / (ray_num - 1) * (float)v) / 180.0 * M_PI)) / cos((double)((-fov / 2.0 + fov / (ray_num - 1) * (float)v) / 180.0 * M_PI)));
		float static_vec_y = (float)(sin((double)((angle - fov / 2.0 + fov / (ray_num - 1) * (float)v) / 180.0 * M_PI)) / cos((double)((-fov / 2.0 + fov / (ray_num - 1) * (float)v) / 180.0 * M_PI)));
		for(int z = 0; z < distance * scene->resolution; z++)
		{
			float vec_x = static_vec_x * (float)(z+1) / (float)(scene->resolution)+camera_pos_x;
			float vec_y = static_vec_y * (float)(z + 1) / (float)(scene->resolution)+camera_pos_y;
			if ((int)(vec_x) >= 0 && (int)(vec_y) && (int)(vec_x) < scene->field_size_x && (int)(vec_y) < scene->field_size_y)
			{
				for (int i = 0; i < scene->resolution * scene->field_size_z; i++)
				{
					for (int rgb = 0; rgb < 3; rgb++)
					{
						set_pixel(result, v, i, z, rgb, get_asset(scene, get_field(scene, (int)vec_x,(int)vec_y,i/ scene->resolution),i% scene->resolution,(int)vec_y* scene->resolution% scene->resolution, (int)vec_x * scene->resolution % scene->resolution, rgb));
					}
				}
			}
		}
	}

	return result;
}

};

//g++ -fPIC -shared -o libtest.so main.cpp