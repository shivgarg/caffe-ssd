// This program converts a set of images and annotations to a lmdb/leveldb by
// storing them as AnnotatedDatum proto buffers.
// Usage:
//   convert_annoset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images and
// annotations, and LISTFILE should be a list of files as well as their labels
// or label files.
// For classification task, the file should be in the format as
//   imgfolder1/img1.JPEG 7
//   ....
// For detection task, the file should be in the format as
//   imgfolder1/img1.JPEG annofolder1/anno1.xml
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <pthread.h>
#include <unistd.h>
#include <sys/sysinfo.h>

#include "boost/scoped_ptr.hpp"
#include "boost/variant.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
    "The backend {lmdb, leveldb} for storing the result");
DEFINE_string(anno_type, "classification",
    "The type of annotation {classification, detection}.");
DEFINE_string(label_type, "xml",
    "The type of annotation file format.");
DEFINE_string(label_map_file, "",
    "A file with LabelMap protobuf message.");
DEFINE_bool(check_label, false,
    "When this option is on, check that there is no duplicated name/label.");
DEFINE_int32(min_dim, 0,
    "Minimum dimension images are resized to (keep same aspect ratio)");
DEFINE_int32(max_dim, 0,
    "Maximum dimension images are resized to (keep same aspect ratio)");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_int32(thread_count, 8, "Thread counts");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");
DEFINE_bool(append, false, "Append to existing db");

typedef struct _LMDBData {
  bool is_color;
  bool encoded;
  string encode_type;
  string anno_type;
  AnnotatedDatum_AnnotationType type;
  string label_type;
  string label_map_file;
  bool check_label;
  std::map<std::string, int> name_to_label;

  int min_dim;
  int max_dim;
  int resize_height;
  int resize_width;
  int offset;

  std::string root_folder;
  std::vector<std::pair<std::string, boost::variant<int, std::string> > > lines;

  db::DB* db;
  int thread_num;
  int thread_idx;
} LMDBData;
int g_count = 0;
pthread_mutex_t mutex_lock;

void *write_to_lmdb(void* _data) {
  LMDBData* data = (LMDBData*)_data;
  AnnotatedDatum anno_datum;
  Datum* datum = anno_datum.mutable_datum();
  db::Transaction* txn = data->db->NewTransaction();
  int step = data->lines.size() / data->thread_num ;
  int from = step * data->thread_idx;
  int to;
  int count = 0;
  if (data->thread_num == data->thread_idx + 1) {
    to = data->lines.size();
  } else {
    to = from + step;
  }
  for (int line_id = from; line_id < to; ++line_id) {
    bool status = true;
    std::string enc = data->encode_type;
    if (data->encoded && !enc.size()) {
      // Guess the encoding type from the file name
      string fn = data->lines[line_id].first;
      size_t p = fn.rfind('.');
      if ( p == fn.npos )
        LOG(WARNING) << "Failed to guess the encoding of '" << fn << "'";
      enc = fn.substr(p);
      std::transform(enc.begin(), enc.end(), enc.begin(), ::tolower);
    }
    std::string filename = data->root_folder + data->lines[line_id].first;
    if (data->anno_type == "classification") {
      int label = boost::get<int>(data->lines[line_id].second);
      status = ReadImageToDatum(filename, label, data->resize_height, data->resize_width,
          data->min_dim, data->max_dim, data->is_color, enc, datum);
    } else if (data->anno_type == "detection") {
      std::string labelname = data->root_folder + boost::get<std::string>(data->lines[line_id].second);
      status = ReadRichImageToAnnotatedDatum(filename, labelname, data->resize_height,
          data->resize_width, data->min_dim, data->max_dim, data->is_color, enc, data->type, data->label_type,
          data->name_to_label, &anno_datum);
      anno_datum.set_type(AnnotatedDatum_AnnotationType_BBOX);
    }
    if (status == false) {
      LOG(WARNING) << "Failed to read " << data->lines[line_id].first;
      continue;
    }
//    if (data->check_size) {
//      if (!data_size_initialized) {
//        data_size = datum->channels() * datum->height() * datum->width();
//        data_size_initialized = true;
//      } else {
//        const std::string& data = datum->data();
//        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
//            << data.size();
//      }
//    }
    // sequential
    string key_str = caffe::format_int(line_id + data->offset, 8) + "_" + data->lines[line_id].first;

    // Put in db
    string out;
    CHECK(anno_datum.SerializeToString(&out));
    txn->Put(key_str, out);

    __atomic_fetch_add(&g_count, 1, __ATOMIC_SEQ_CST);
    if (++count % 1000 == 0) {
      // Commit db
      pthread_mutex_lock(&mutex_lock);
      txn->Commit();
      pthread_mutex_unlock(&mutex_lock);
      delete txn;
      txn = data->db->NewTransaction();
      LOG(INFO) << "Processed " << g_count << " files.";
    }
  }

  if (count % 1000 != 0) {
    pthread_mutex_lock(&mutex_lock);
    txn->Commit();
    pthread_mutex_unlock(&mutex_lock);
    LOG(INFO) << "Processed " << g_count << " files.";
  }
}

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images and annotations to the "
        "leveldb/lmdb format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_annoset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_annoset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;
  const string anno_type = FLAGS_anno_type;
  AnnotatedDatum_AnnotationType type;
  const string label_type = FLAGS_label_type;
  const string label_map_file = FLAGS_label_map_file;
  const bool check_label = FLAGS_check_label;
  std::map<std::string, int> name_to_label;
  const int thread_num = FLAGS_thread_count;
  const bool append = FLAGS_append;
  std::ifstream infile(argv[2]);
  std::vector<std::pair<std::string, boost::variant<int, std::string> > > lines;
  std::string filename;
  int label;
  std::string labelname;

  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  int start_line_id = 0;
  if (append) {
      db->Open(argv[3], db::WRITE);
      scoped_ptr<db::Cursor> cursor(db->NewCursor());
      while (cursor->valid()) {
          start_line_id++;
          cursor->Next();
      }
      // start from the next line_id
      start_line_id++;
      LOG(INFO) << "Append from line " << start_line_id << ".";
  } else {
    db->Open(argv[3], db::NEW);
  }



  if (anno_type == "classification") {
    while (infile >> filename >> label) {
      lines.push_back(std::make_pair(filename, label));
    }
  } else if (anno_type == "detection") {
    type = AnnotatedDatum_AnnotationType_BBOX;
    LabelMap label_map;
    CHECK(ReadProtoFromTextFile(label_map_file, &label_map))
        << "Failed to read label map file.";
    CHECK(MapNameToLabel(label_map, check_label, &name_to_label))
        << "Failed to convert name to label.";
    int cnt = start_line_id - 1;
    while (infile >> filename >> labelname) {
      if (cnt > 0) {
        cnt--;
        continue;
      }
      lines.push_back(std::make_pair(filename, labelname));
    }
  }

  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int min_dim = std::max<int>(0, FLAGS_min_dim);
  int max_dim = std::max<int>(0, FLAGS_max_dim);
  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Storing to db
  std::string root_folder(argv[1]);

  pthread_t threads[thread_num];
  LMDBData lmdbData[thread_num];
  int i = 0;
  for (i = 0; i < thread_num; ++i) {
    lmdbData[i].is_color = is_color;
    lmdbData[i].encoded = encoded;
    lmdbData[i].encode_type = encode_type;
    lmdbData[i].anno_type = anno_type;
    lmdbData[i].type = type;
    lmdbData[i].label_type = label_type;
    lmdbData[i].label_map_file = label_map_file;
    lmdbData[i].check_label = check_label;
    lmdbData[i].name_to_label = name_to_label;
    lmdbData[i].min_dim = min_dim;
    lmdbData[i].max_dim = max_dim;
    lmdbData[i].resize_height = resize_height;
    lmdbData[i].resize_width = resize_width;
    lmdbData[i].offset = start_line_id;
    lmdbData[i].root_folder = root_folder;
    lmdbData[i].lines = lines;
    lmdbData[i].db = db.get();
    lmdbData[i].thread_num = thread_num;
  }

  pthread_mutex_init(&mutex_lock, NULL);
  for (i = 0; i < thread_num; ++i) {
    lmdbData[i].thread_idx = i;
    pthread_create(&threads[i], NULL, write_to_lmdb, &lmdbData[i]);
  }
  for (i = 0; i < thread_num; ++i)
    pthread_join(threads[i], NULL);
  pthread_mutex_destroy(&mutex_lock);

#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}

