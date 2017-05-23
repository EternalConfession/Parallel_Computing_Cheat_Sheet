# Parallel_Computing_Cheat_Sheet
In case I forget ...

Parallel Computing Cheat Sheet.

This is a cheat sheet for parallel computing.
In case I forgot what I learned during the course when I need to use parallel computing tools, though I hardly went to class.

1. MPI

1.1 Intro

    MPI is multi-process. It is SPMD (Single Program Multiple Data)
    It launch several processes after compiling.
    Use MPI_COMMON_WORLD as a communicator, and each process has a MPI_COMMON_WORLD after launching.

1.2 Basic structured

    Use if - else structure to split tasks.

    if (my_rank == 0) {
        ...
    } else {
        ...
    }

    Use MPI_SEND & MPI_REV to do communication.
    The data_buffer, buffer_size, data_type, dest/source, data_tag, MPI_Common need to be defined when using these functions._Advance
    
1.3 Input Data Rule

    Usually, we use(only allow) process_0 to interact with stdin

    if (my_rank == 0) {
        read data from Users...
        send data to other processes... (Broadcast) //You can also scatter the data, if you know the structure of the data and how to split it clearly
    } else {
        recv_data...
    }

1.4 Collective Communication

    (1) Tree structured communication : Recursively pass data to your (for example) right neighbor.

    (2) MPI_Reduce:
    
    We have discussed common communications between two process. Now, lets look at collective communication. It is kind of like Reduce.

    Use the API: 
        
            MPI_REDUCE(void* &input_data, void* &output_data, int count, MPI_DATA_TYPE type, MPI_OP operator, int dest, MPI_Comm comm)

    Some Predefined Operator:

            MPI_MAX, MPI_MIN, MPI_SUM, MPI_PROD, MPI_LAND, MPI_BAND, MPI_LOR, MPI_BOR, MPI_MAXLOC, MPI_MINLOC

            MINLOC & MAXLOC are very important when find minimum ... with its index.
    
    All the processes need to set one Dest.

    We also need to define the output_data in each process..even if its NULL

    (3) MPI_Allreduce：

    When each process need the global sum

    First Reduce to Dest. Then the dest will distribute the data. 

    (4) MPI_Bcast(void* data, int count, MPI_DATA_TYPE type, int source, MPI_Comm)

    BroadCast Data to all processes._Advance


*************************************************************************************************************************
When we do Scatter, BroadCast, Send & Recv etc.  We need to alloc memory.
... When you call these functions, you have already define the size._Advance
When we do reduce ... do we need to alloc data for output_data in every process???
*************************************************************************************************************************

1.5 Scatter & Gather

Distribute the Data to each process... Each require a peice of data.

For example, distribute a vector. 

Scatter(void* send_buffer, int send_count, MPI_DATA_TYPE, void* recv_buffer, int recv_count, MPI_DATA_TYPE, int, MPI_Comm)
Gather(...) //Just in the reverse orde
Allgather(...)  Like all reduce
//MPI_SEND, REDUCE, Scatter, Gather are sychronized themselves..

1.6 Derived Data Type

//MPI_DATA_Type is the class of derived data type.

MPI_Type_Create_Struct(int count, int array_of_block_length[], int MPI_Aint array_of_displacement[], MPI_Data_Type array_of_dataType[], MPI_Data_Type* new_type)

//Get the memory address of a location..Use to create struct.
int MPI_Get_Address(void *location, MPI_AInt* address)
//After create Struct, we need commit Type
int MPI_Type_commit();
int MPI_Type_free();


1.7 MPI_Barrier

Make Sure that every process has done its work...(For example..Time calcullation)

Look back later...



2. Pthread

2.1 Intro

Pthread is shared memory. We need to cotrol access to critical section._Advance

Some troublesome things : Thread synchronize, mutex, producer-cosumer synchronization & semaphores, barrier & condition variable, lock...

2.2 Basic structure

long thread_numbers;
pthread_t* thread_handles;

thread_handles = malloc (thread_numbers * sizeof(pthread_t));
for each thread
    pthread_create(thread, NULL, function, input..(Usually thread id));
...
for each thread
    pthread_join(thread, NULL);

free(handle)


2.3 Pthread_Dijkstra
//Refer to assignments...lol
//...The p_thread program Analysis...

Each Thread share memory...So they do not need to communicate...While we need to care about critical areas..So it's the mean difference between MPI & Pthread or..difference between shared or multiple memory.

struct thread_parameter = {int start, end, my_id;};

void* thread_Work(void* para)
{
    for (int phase = 0; phase < N; phase++)
    {
        thread_parameter my_para = *(thread_parameter*) para;
        for (int i = my_para.start; i < my_para.end; i++)
        {
            find local minimum;
            (loc_min = min;
            loc_index = i;)
        }
        min_index[my_para.id] = i;
        min_dist[my_para.id] = loc_min;

        //Critical Area...This area is excuted thread by thread...
        Find global min & set global min at the last threads...

        //Update each part..
        local update...
    }
}

2.4 Critical Area

When multiple threads want to write to same position. We need to let them enter the critical area one by one.

(1) Busy-Waiting：

    Use a while loop...

    while (flag != my_rank);
    do somethind;
    flag = (flag+1)%flag;

    Disadvantages: Use CPU but do nothing.

(2) Mutex:

    A special type of variable that can restrict the access to a certain area ... 
   
    pthread_mutex_t global_mutex;
    intit_mutex(&global_mutex, NULL);
    ...
    when you need to use it, lock...unlock...
    
    Except access to a critical area one by one ... can be viewed as single... 

2.5 Producer-Consumer synchronization and semaphores

    (1) Semaphores
        
        Busy-waiting enforces the order..(one by one)
        Mutex leave the order to chance..
        How can we control the order..
        Actually its synchronization...We can also view it as how many resources left...

        Example : Message Sending...(post = +1;  wait = -1)
        wait on dest, post itself...
    
    (2) Barrier & Condition Variable

        all threads have done..

        two ways to implement barrier.

        (2.1) Use mutex;
        //...(2.2) Use semaphores;  //Not yet look

    (3) Condition Variable

        Allows thread suspend until a certain event...

        (2.3) implement a barrier with condition variable

        Condition variables are always used in conjunction with a mutex, and have three operations: 
        
        wait(m): unlocks the mutex m, then blocks the calling thread until some other thread signals the condition.
        
        signal: Wakes up one thread waiting on this condition.
        
        broadcast: Wakes up all threads waiting on this condition.





Need to review:

    The first several chapters : some definition & computing

    CUDA architechture & CUDA Program Basic;

    CUDA Scatter & Reduce.

6-8 hours...
