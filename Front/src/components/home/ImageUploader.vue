<template>
    <div class="uploader"
        @dragenter="OnDragEnter"
        @dragleave="OnDragLeave"
        @dragover.prevent
        @drop="onDrop"
        :class="{ dragging: isDragging }">
        
        <div class="upload-control" v-show="images.length">
            <div class="d-flex justify-content-end">                
                <label class="ml-8" for="file">Select a file</label>
                <button @click="upload(typeId);">Upload</button>
            </div>
            <!-- <blog-post v-bind:comment-ids="post.commentIds"></blog-post> -->
            <!-- <testbook :submittedFname="submittedFname" :submittedLname="submittedLname" ></testbook> -->
        </div>


        <div v-show="!images.length">
            <i class="fa fa-cloud-upload"></i>
            <p>Drag your images here</p>
            <div>OR</div>
            <div class="file-input">
                <label for="file">Select a file</label>
                <input type="file" id="file" @change="onInputChange" multiple ref="myFileInput"/>
            </div>
        </div>

        <div class="images-preview" v-show="images.length">

            <div class="img-wrapper" v-for="(image, index) in images" :key="index">
                <img :src="image" :alt="`Image Uplaoder ${index}`">
                <div class="details">
                    <div class="name" style="word-break:break-all;" v-text="files[index].name"></div>
                    <span class="size" v-text="getFileSize(files[index].size)"></span>
                </div>
            </div>
        </div>

    </div>
</template>

<script>
import axios_common from '../../axios_common';
import { mapGetters } from 'vuex';
// import result from "../book/result"
import bus from '../../utiles/bus.js'


export default {
    props: ['info'],
    data: () => ({
        loadingStatus: false,
        isDragging: false,
        dragCount: 0,
        files: [],
        images: [],
        texts:[],
        imagesrc:[],
        formData:[],
        openstate: false,
        typeId: 1,
    }),
    computed: {
        ...mapGetters([
            'isAuthenticated',
            'requestHeader',
            'userId',
            'username'
        ])
    },
    created() {
        this.openstate =this.info;
    },
    components: {
        // testbook
    },
    methods: {
        OnDragEnter(e) {
            e.preventDefault();
            
            this.dragCount++;
            this.isDragging = true;
            return false;
        },
        OnDragLeave(e) {
            e.preventDefault();
            this.dragCount--;
            if (this.dragCount <= 0)
                this.isDragging = false;
        },
        onInputChange(e) {
            const files = e.target.files;
            Array.from(files).forEach(file => this.addImage(file));
        },
        onDrop(e) {
            e.preventDefault();
            e.stopPropagation();
            this.isDragging = false;
            const files = e.dataTransfer.files;
            Array.from(files).forEach(file => this.addImage(file));
        },
        addImage(file) {
            if (!file.type.match('image.*')) {
                console.log("이미지 파일이 아닌데영");
                return;
            }
            this.files.push(file);
            const reader = new FileReader();
            reader.onload = (e) => this.images.push(e.target.result);
            reader.readAsDataURL(file);

        },
        getFileSize(size) {
            const fSExt = ['Bytes', 'KB', 'MB', 'GB'];
            let i = 0;
            
            while(size > 900) {
                size /= 1024;
                i++;
            }
            return `${(Math.round(size * 100) / 100)} ${fSExt[i]}`;
        },
        clear(files){
            this.files.forEach(function(value, index){

                console.log(value +","+ index + "," +files)
                value = '';
            })
        },
        upload(id) {
            const formData = new FormData();
            this.files.forEach(file => {
                formData.append('images[]', file, file.name);
            });     
            bus.$emit('start:loading')
            console.log(id)
            setTimeout(() => {
                axios_common.post(`/sub3/ImageUpload/${parseInt(id)}/`, formData, this.requestHeader)
                    .then(response => {
                        // console.log("---------------------")
                        this.imagesrc = response.data.image;
                        this.texts = response.data.text;
                        console.log(this.texts)
                        // console.log(this.imagesrc)
                        // console.log(this.texts)
                        console.log(response.data)
                        this.$router.push({ name: 'result', params: {response: response.data}})
                        bus.$emit('end:loading');
                    })
                    .catch(error => console.log(error))
            }, 3000);
            //이미지 배열 0 으로 초기화(이미지 내리는 용도)
            this.images.splice(0)
            this.files.splice(0)

            //모달 창 닫히도록
            this.openstate= false;
            this.$emit('child', this.openstate)
          
            
        },

    }
}
</script>

<style lang="scss" scoped>
.uploader:onclick {
    display: hidden;
}
.uploader {
    width: 100%;
    background: #2196F3;
    color: #fff;
    padding: 40px 15px;
    text-align: center;
    border-radius: 10px;
    border: 3px dashed #fff;
    font-size: 20px;
    position: relative;
    &.dragging {
        background: #fff;
        color: #2196F3;
        border: 3px dashed #2196F3;
        .file-input label {
            background: #2196F3;
            color: #fff;
        }
    }
    i {
        font-size: 85px;
    }
    .file-input {
        width: 200px;
        margin: auto;
        height: 68px;
        position: relative;
        label,
        input {
            background: #fff;
            color: #2196F3;
            width: 100%;
            position: absolute;
            left: 0;
            top: 0;
            padding: 10px;
            border-radius: 4px;
            margin-top: 7px;
            cursor: pointer;
        }
        input {
            opacity: 0;
            z-index: -2;
        }
    }
    .images-preview {
        display: flex;
        flex-wrap: wrap;
        margin-top: 20px;
        .img-wrapper {
            width: 160px;
            display: flex;
            flex-direction: column;
            margin: 10px;
            height: 150px;
            justify-content: space-between;
            background: #fff;
            box-shadow: 5px 5px 20px #3e3737;
            img {
                max-height: 105px;
            }
        }
        .details {
            font-size: 12px;
            background: #fff;
            color: #000;
            display: flex;
            flex-direction: column;
            align-items: self-start;
            padding: 3px 6px;
            .name {
                overflow: hidden;
                height: 18px;
            }
        }
    }
    .upload-control {
        position: absolute;
        width: 100%;
        background: #fff;
        top: 0;
        left: 0;
        border-top-left-radius: 7px;
        border-top-right-radius: 7px;
        padding: 10px;
        padding-bottom: 4px;
        text-align: right;
        button, label {
            background: #2196F3;
            border: 2px solid #03A9F4;
            border-radius: 3px;
            color: #fff;
            font-size: 15px;
            cursor: pointer;
        }
        label {
            padding: 2px 5px;
            margin-right: 10px;
        }
    }
}
</style>
